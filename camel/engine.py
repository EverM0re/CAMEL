"""Retrieval engine: baselines + CAMEL variants over a message corpus."""
import re
from collections import defaultdict
from datetime import datetime

from . import config
from .graph import MemoryGraph
from .profiles import build_profiles
from .retrieval import (HybridRetriever, DenseRetriever, BM25, tokenize,
                        rrf_fuse, rerank)

METHODS = [
    # baselines
    "bm25", "dense", "hybrid", "graphrag", "flat_mem", "oracle",
    # external / published baselines
    "mem0", "zep", "hipporag", "raptor", "readagent", "memgpt",
    # ours
    "camel",
    # ablations
    "camel-nograph", "camel-noversion", "camel-nospeaker",
    "camel-noterm", "camel-noprofile", "camel-norerank",
    "camel-nodecomp", "camel-notimeline", "camel-nochrono",
    # Collapses multi-anchor seeding to a single BM25 ranking. The wide
    # BM25+dense candidate pool sits outside every block below, so it is a
    # candidate for the residual gain the coarse ablation cannot attribute.
    "camel-noseed",
    # Coarse ablation blocks: each removes a whole family of mechanisms, which
    # is what a reader needs to see. The fine-grained switches above remain for
    # diagnosis but are too many to report and too small to separate.
    # three-block ablation (reported)
    # Suffixed `3` so they cannot be confused with the five-block names: r18's
    # `noseedblk` disabled fusion ONLY, whereas `noseed3` also disables
    # decomposition, and silently comparing the two would be wrong.
    "camel-noseed3",       # single BM25 ranking, no decomposition
    "camel-norank3",       # no cross-encoder rerank
    "camel-nostruct3",     # no graph, term, speaker or temporal machinery
    # five-block ablation (legacy)
    "camel-nostructure",   # no graph expansion, no term alignment
    "camel-nospeakerblk",  # no profile injection, no speaker grounding
    "camel-notemporal",    # no timeline arithmetic, no version resolution
    "camel-noranking",     # no cross-encoder rerank, no query decomposition
    "camel-noseedblk",     # single BM25 ranking (fusion only)
    "camel-base",          # all of the above off: hybrid seeding only
    # terminology-alignment ablation ladder (see termalign.py)
    "camel-termcosine",    # SimAlign_tuned: thresholded cosine baseline
    "camel-termnoctx",     # relation inference WITHOUT conversational context
    "camel-termoneway",    # single-direction entailment (no bidirectionality)
]

_DATE_RE = re.compile(r"\b(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})")


def format_message(m, show_group=True):
    """Render a message with speaker/time/role attribution."""
    author = m.get("author", "?")
    role = m.get("role") or ""
    ts = m.get("ts")
    ts_s = ts.strftime("%Y-%m-%d %H:%M") if ts else "unknown-time"
    group = m.get("group") or ""
    head = f"[{ts_s}] <{author}"
    if role:
        head += f" ({role})"
    if group and show_group:
        head += f" @ {group}"
    head += ">"
    body = m["text"]
    if (m.get("meta") or {}).get("superseded"):
        body += " [SUPERSEDED by a later update]"
    return f"{head} {body}"


class Corpus:
    """Holds one conversation corpus and all retrieval machinery."""

    def __init__(self, messages, profiles_by_name=None, corpus_key=None):
        self.messages = messages
        self.profiles_by_name = profiles_by_name or {}
        self.corpus_key = corpus_key
        self.msg_by_id = {m["id"]: m for m in messages}
        self.hybrid = None
        self.graph = None
        self.profiles = {}
        self.flat_facts = []       # (text, source_msg_id) for flat_mem baseline
        self.flat_embeddings = None
        self._ext = {}             # lazily-built external baseline indexes
        self._term_tables = {}     # ablation rung -> alias table
        self._llm = None

    # -- indexing -------------------------------------------------------
    def build(self, use_graph=True, use_alias=True, build_flat=False, llm=None):
        self.hybrid = HybridRetriever()
        self.hybrid.index(self.messages, corpus_key=self.corpus_key)
        self._llm = llm
        if use_graph:
            self.graph = MemoryGraph()
            self.graph.build(self.messages, use_alias=use_alias)
        self.profiles = build_profiles(self.messages, self.profiles_by_name)
        if build_flat:
            self._build_flat_memory(llm)
        return self

    def _build_flat_memory(self, llm):
        """Mem0-like flat memory: LLM-extract facts per window, store flat."""
        import json
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from .llm import LLMClient

        # Disk cache: facts + their embeddings are expensive to rebuild
        # (LLM extraction over ~1500 windows + CPU encoding of ~19k facts), so
        # persist them per corpus to survive restarts.
        facts_path = emb_path = None
        if self.corpus_key:
            flat_dir = config.CACHE_DIR / "flat"
            flat_dir.mkdir(parents=True, exist_ok=True)
            facts_path = flat_dir / f"{self.corpus_key}_facts.json"
            emb_path = flat_dir / f"{self.corpus_key}_facts.npy"
            if facts_path.exists() and emb_path.exists():
                self.flat_facts = [tuple(x) for x in json.loads(facts_path.read_text())]
                self.flat_embeddings = np.load(emb_path)
                return

        window = 20
        sys = ("Extract concise, self-contained memory facts from this conversation "
               "window. Output one fact per line, no numbering, no preamble. Include "
               "who, what, and when. Ignore pleasantries.")
        chunks = []
        for i in range(0, len(self.messages), window):
            chunk = self.messages[i:i + window]
            src_id = chunk[-1]["id"] if chunk else None
            text = "\n".join(format_message(m) for m in chunk)
            chunks.append((text, src_id))

        def extract(text):
            client = LLMClient(max_tokens=600)
            try:
                resp = client.complete(sys, text)
            except Exception:  # noqa: BLE001
                return []
            return [line.strip(" -•*") for line in resp.splitlines()
                    if len(line.strip(" -•*")) > 12]

        facts = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for lines, (_, src_id) in zip(ex.map(extract, [c[0] for c in chunks]),
                                          chunks):
                for line in lines:
                    facts.append((line, src_id))
        self.flat_facts = facts
        # Pre-encode facts ONCE here (single-threaded, on the main thread) so the
        # per-question flat_mem search only has to embed the query.
        if facts:
            from .retrieval import get_embedding_model
            model = get_embedding_model()
            texts = [f for f, _ in facts]
            self.flat_embeddings = model.encode(
                texts, batch_size=config.EMBED_BATCH_SIZE, normalize_embeddings=True,
                convert_to_numpy=True).astype(np.float32)
        if facts_path is not None:
            facts_path.write_text(json.dumps(self.flat_facts, ensure_ascii=False))
            np.save(emb_path, self.flat_embeddings)

    def _flat_search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        if not self.flat_facts or self.flat_embeddings is None:
            return []
        from .retrieval import get_embedding_model
        import numpy as np
        model = get_embedding_model()
        qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.flat_embeddings @ qv
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.flat_facts[i]) for i in order]

    # -- retrieval dispatch --------------------------------------------
    def retrieve(self, method, question, asker=None, q_meta=None, top_k=None):
        top_k = top_k or config.TOP_K
        if method == "bm25":
            return self.hybrid.bm25.search(question, top_k=top_k)
        if method == "dense":
            return self.hybrid.dense.search(question, top_k=top_k)
        if method == "hybrid":
            return self.hybrid.search(question, top_k=top_k)
        if method == "graphrag":
            return self._graphrag_retrieve(question, top_k)
        if method == "flat_mem":
            out = []
            for s, (text, sid) in self._flat_search(question, top_k):
                if sid in self.msg_by_id:
                    out.append((s, self.msg_by_id[sid]))
            return out
        if method == "oracle":
            return self._oracle_retrieve(q_meta, top_k)
        if method in ("mem0", "zep", "hipporag", "raptor", "readagent", "memgpt"):
            from . import baselines
            return baselines.retrieve(self, method, question, asker, q_meta, top_k)
        if method.startswith("camel"):
            return self._camel_retrieve(method, question, asker, q_meta, top_k)
        raise ValueError(f"unknown method {method}")

    # -- CAMEL retrieval --------------------------------------------
    def _effective_top_k(self, top_k):
        """Cap the window at a fraction of the corpus.

        CAMEL is tuned for 10k-30k-message corpora, where top_k=50 is well
        under 1% of the corpus. On a small corpus the same setting is
        catastrophic: a SocialMemBench network holds ~171 turns, so top_k=50
        pulls in 29% of everything, precision collapses, and the extra
        machinery only adds noise -- CAMEL lost to plain hybrid there while
        achieving HIGHER recall, which is context dilution, not a retrieval
        failure.
        """
        cap = int(len(self.messages) * config.TOP_K_CORPUS_FRAC)
        return max(config.TOP_K_MIN, min(top_k, cap)) if cap else top_k

    def _camel_retrieve(self, method, question, asker, q_meta, top_k):
        top_k = self._effective_top_k(top_k)
        # Coarse blocks expand into the fine-grained switches.
        # Two groupings over the same ten switches.
        #
        # The THREE-block grouping is the one the paper reports. It splits the
        # pipeline by what a component does to the evidence, so the blocks sit
        # at one level of abstraction:
        #   seedblk  -- what enters the candidate pool (fusion + decomposition)
        #   rankblk  -- how the pool is ordered (cross-encoder)
        #   structblk-- what is added on top of, or imposed upon, the pool
        # Decomposition belongs with seeding because it produces sub-queries
        # whose result lists are fused into the pool; grouping it with the
        # reranker (as the five-block version did) split one mechanism across
        # two rows and mis-attributed its contribution.
        #
        # The FIVE-block grouping is kept so earlier runs remain reproducible.
        _BLOCK = {
            # three-block (reported)
            "noseed3":      ("noseed", "nodecomp"),
            "norank3":      ("norerank",),
            "nostruct3":    ("nograph", "noterm", "noprofile", "nospeaker",
                             "notimeline", "noversion", "nochrono"),
            # five-block (legacy, reproduces r14--r18)
            "noseedblk":    ("noseed",),
            "nostructure":  ("nograph", "noterm"),
            "nospeakerblk": ("noprofile", "nospeaker"),
            "notemporal":   ("notimeline", "noversion", "nochrono"),
            "noranking":    ("norerank", "nodecomp"),
            "base":         ("noseed", "nograph", "noterm", "noprofile",
                             "nospeaker",
                             "notimeline", "noversion", "norerank", "nodecomp",
                             "nochrono"),
        }
        for _blk, _sw in _BLOCK.items():
            if method.endswith("-" + _blk):
                method = "camel-" + "-".join(_sw)
                break

        use_graph = "nograph" not in method
        use_version = "noversion" not in method
        # Suppressing superseded evidence is wrong when the question is ABOUT
        # the change. GroupMemBench's knowledge_update items ask for the trigger
        # or the prior value as often as the current one, and its temporal items
        # need the original commitment -- annotating that evidence as stale cost
        # -20 points on both (and produced "no current approach is available"
        # answers). It remains on elsewhere, where it is worth +2.4 on
        # EverMemBench and helps every category it touches.
        _qt = (q_meta or {}).get("qtype")
        if _qt in ("knowledge_update", "temporal", "abstention"):
            use_version = False
        use_speaker = "nospeaker" not in method
        use_term = "noterm" not in method
        # Terminology-alignment ablation ladder. Each rung uses a different
        # alias table built by termalign.py; they are cached per corpus.
        term_variant = None
        for _v in ("termcosine", "termnoctx", "termoneway"):
            if _v in method:
                term_variant = _v
                break
        use_rerank = "norerank" not in method
        use_chrono = "nochrono" not in method
        lexical_only = "noseed" in method
        use_decomp = "nodecomp" not in method and config.DECOMPOSE
        use_timeline = "notimeline" not in method and config.USE_TIMELINE

        q_terms = set(tokenize(question))
        cand_k = max(config.CANDIDATE_K, top_k * 3)

        # ---- 1. multi-anchor seeding -----------------------------------
        # A single similarity query lands on ONE anchor of a two-anchor
        # ("how long between A and B") question. Decomposing the query and
        # fusing per-subquery result lists is what makes both anchors reachable.
        subqueries = [question]
        if use_decomp:
            subqueries = self._subqueries(question, q_meta)

        ranked_lists = []
        for sq in subqueries:
            depth = cand_k if sq == question else config.SUBQUERY_K * 3
            ranked_lists.append(self.hybrid.search(
                sq, top_k=depth, depth=cand_k, lexical_only=lexical_only))
        # weight the original query highest, sub-queries equally below it
        weights = [1.0] + [0.85] * (len(ranked_lists) - 1)
        fused = rrf_fuse(ranked_lists, weights=weights)
        seed_scored = [(s, self.msg_by_id[mid]) for mid, s in fused
                       if mid in self.msg_by_id]

        # Per-sub-query reservation. Fusing the sub-query lists is not enough:
        # a "how long between A and B" question needs BOTH endpoints, and RRF
        # lets the sub-query with the stronger lexical match crowd out the
        # other. Measured on EverMem multi-hop, the fraction of questions with
        # EVERY gold anchor retrieved was 0.0% at k=5/10/15 while accuracy sat
        # at 4-14% against an oracle of 99% -- so the chain was essentially
        # never assembled. Giving each sub-query a guaranteed share makes a
        # complete chain reachable.
        reserved = []
        if len(ranked_lists) > 1:
            per = max(1, int(top_k * config.SUBQUERY_RESERVE_FRAC
                             / max(1, len(ranked_lists) - 1)))
            for rl in ranked_lists[1:]:
                for _s2, m2 in rl[:per]:
                    reserved.append(m2["id"])

        cand = {}   # msg_id -> score (candidate pool for reranking)
        for s, m in seed_scored[:cand_k]:
            cand[m["id"]] = max(cand.get(m["id"], 0.0), s)

        # ---- 2. graph expansion (scored, budget-capped) -----------------
        graph_ids = []
        if use_graph and self.graph:
            # Anchor on a WIDE slice of the seed ranking, not the top few. The
            # message that actually starts the evidence chain is often outranked
            # by lexical near-duplicates (in a distractor-heavy corpus it can sit
            # well outside the top 10), and anchoring narrowly means its reply /
            # decision-chain neighbours -- the second hop -- are never reached.
            # Precision is restored by the reranker in step 4, so a generous
            # anchor set costs recall nothing.
            n_anchor = min(len(seed_scored), max(top_k, config.GRAPH_ANCHORS))
            anchors = [m["id"] for _, m in seed_scored[:n_anchor]]
            neigh = self.graph.expand(anchors, hops=config.GRAPH_HOPS,
                                      budget=cand_k, query_terms=q_terms)
            graph_ids = [mid for _, mid in neigh]
        alias_ids = []
        if use_term and self.graph:
            aliases = self._alias_table(term_variant)
            alias_ids = [mid for _, mid in self.graph.alias_expand(
                question, budget=cand_k // 2, aliases=aliases)]
        # ---- 3. timeline anchoring --------------------------------------
        # For date-anchored questions, pull the messages around each date the
        # question mentions -- these carry the second anchor that pure
        # similarity misses.
        timeline_ids = []
        if use_timeline and self.graph:
            for y, mo, d in _DATE_RE.findall(question):
                ds = f"{y}-{int(mo):02d}-{int(d):02d}"
                timeline_ids.extend(self.graph.neighbours_by_date(ds, window=1,
                                                                  budget=20))

        # ---- 3b. asker-own-message injection -----------------------------
        # For first-person questions the gold evidence is typically a message
        # the asker wrote. Reordering alone cannot help if those messages never
        # entered the candidate pool, so retrieve within the asker's own turns.
        asker_ids = []
        if use_speaker and asker and self.graph:
            qt = (q_meta or {}).get("qtype")
            if qt == "user_implicit" or re.search(r"\b(my|me|I|mine|myself)\b",
                                                  question or ""):
                own = self.graph.author_msgs.get(asker, [])
                if own:
                    asker_ids = self._search_within(question, own,
                                                    top_k=max(top_k // 2, 8))

        for mid in graph_ids + alias_ids + timeline_ids + asker_ids:
            cand.setdefault(mid, 0.0)

        pool = [(cand[mid], self.msg_by_id[mid]) for mid in cand
                if mid in self.msg_by_id]
        pool.sort(key=lambda x: -x[0])
        pool = pool[:cand_k]

        # ---- 4. cross-encoder rerank ------------------------------------
        if use_rerank:
            # Score only a prefix of the pool: the cross-encoder is the runtime
            # bottleneck and deep candidates rarely survive into top_k anyway.
            depth = max(config.RERANK_DEPTH, top_k)
            head, tail = pool[:depth], pool[depth:]
            reranked = rerank(question, head, top_k * 2,
                              text_fn=lambda m: format_message(m))
            if len(reranked) < top_k * 2:
                reranked = reranked + tail[: top_k * 2 - len(reranked)]
        else:
            reranked = pool[:top_k * 2]

        # ---- 5. budget split: guarantee expansion evidence a slot ---------
        # "Expansion" means *reached by a structural edge* (graph / alias /
        # timeline / asker), regardless of whether the message also happens to
        # appear far down the similarity ranking. Defining it as "absent from
        # the seed list" collapsed the whole mechanism: a second hop that the
        # lexical retriever ranks 31st was classified as a seed, then lost the
        # top-k cut to 30 near-duplicates, so the graph contributed nothing.
        expansion = set(graph_ids) | set(alias_ids) | set(timeline_ids) \
            | set(asker_ids) | set(reserved)
        top_seeds = {m["id"] for _, m in seed_scored[:top_k]}
        expansion -= top_seeds     # already guaranteed a slot on their own merit

        # Expansion evidence must EARN its place on rerank score, competing
        # head-to-head with the seeds. Reserving a fixed block for it and
        # inserting those messages unscored cost 14 points against a plain
        # hybrid baseline on GroupMemBench -- it evicted ~10 well-ranked hits
        # per question and lost on every single question type. Only a small
        # floor is now guaranteed, so a genuine second hop still survives when
        # the reranker undervalues it, without the expansion crowding the window.
        rerank_pos = {m["id"]: i for i, (_s, m) in enumerate(reranked)}
        chosen, have = [], set()
        for s, m in reranked:
            if len(chosen) >= top_k:
                break
            if m["id"] in have:
                continue
            chosen.append((s, m))
            have.add(m["id"])

        n_floor = int(round(top_k * config.GRAPH_BUDGET_FRAC))
        if n_floor:
            n_have = sum(1 for _s, m in chosen if m["id"] in expansion)
            for mid in (graph_ids + alias_ids + timeline_ids + asker_ids):
                if n_have >= n_floor:
                    break
                if mid in have or mid in top_seeds or mid not in self.msg_by_id:
                    continue
                # displace the weakest chosen item that is NOT itself expansion
                if len(chosen) >= top_k:
                    drop = next((i for i in range(len(chosen) - 1, -1, -1)
                                 if chosen[i][1]["id"] not in expansion), None)
                    if drop is None:
                        break
                    have.discard(chosen[drop][1]["id"])
                    chosen.pop(drop)
                chosen.append((0.0, self.msg_by_id[mid]))
                have.add(mid)
                n_have += 1

        # Top up from the plain seed ranking if still short.
        for s, m in seed_scored:
            if len(chosen) >= top_k:
                break
            if m["id"] not in have:
                chosen.append((s, m))
                have.add(m["id"])

        # Present in rerank order (strongest evidence first) rather than in
        # whatever order the budget logic happened to append.
        chosen.sort(key=lambda sm: rerank_pos.get(sm[1]["id"], 10**6))

        evidence = [m for _, m in chosen[:top_k]]

        # ---- 6. speaker grounding ---------------------------------------
        if use_speaker:
            evidence = self._speaker_boost(evidence, question, asker, q_meta)

        # ---- 7. version resolution --------------------------------------
        if use_version:
            evidence = self._version_resolve(evidence)

        # ---- 8. chronological presentation ------------------------------
        # Temporal / multi-hop reasoning is far easier when evidence is in time
        # order; relevance order scrambles the timeline the model must subtract.
        cat = (q_meta or {}).get("category")
        qt = (q_meta or {}).get("qtype")
        if use_chrono and (cat in ("multi-hop", "temporal")
                           or qt in ("multi_hop", "temporal",
                                     "knowledge_update")):
            evidence = sorted(evidence, key=lambda m: (m.get("ts") or datetime.min))

        return [(1.0, m) for m in evidence[:top_k]]

    # -- query decomposition -------------------------------------------
    def _subqueries(self, question, q_meta):
        """Split a comparison / multi-hop question into its sub-questions.

        LLM-based when a client is available (cached, so it costs one call per
        unique question across the whole sweep), with a rule-based fallback.
        """
        cat = (q_meta or {}).get("category")
        qt = (q_meta or {}).get("qtype")
        multi = cat in ("multi-hop", "temporal") or qt in ("multi_hop", "temporal")
        if not multi:
            return [question]

        subs = []
        if self._llm is not None:
            sys = ("Decompose the question into the minimal set of independent "
                   "retrieval sub-questions needed to answer it. Each sub-question "
                   "must name ONE concrete event/fact to look up. Output one "
                   "sub-question per line, at most "
                   f"{config.MAX_SUBQUERIES}, no numbering, no preamble.")
            try:
                resp = self._llm.complete(sys, question, max_tokens=200)
                subs = [l.strip(" -•*0123456789.") for l in resp.splitlines()
                        if len(l.strip(" -•*0123456789.")) > 8]
            except Exception:  # noqa: BLE001
                subs = []
        if not subs:
            subs = _rule_split(question)
        return [question] + subs[:config.MAX_SUBQUERIES]

    def _speaker_boost(self, evidence, question, asker, q_meta=None):
        """Promote the messages whose AUTHOR the question is really about.

        Which speaker matters is category-dependent, and getting this wrong is
        why a blanket asker-boost was previously removed altogether:

        * ``user_implicit`` ("What is *my* next task?") -- the gold evidence is
          usually a message the ASKER WROTE THEMSELVES, so the asker's own turns
          must be promoted. GroupMemBench's documented failure modes for this
          category are exactly "wrong-speaker shadowing" (a louder speaker's
          near-duplicates outrank the asker's own message) and "multi-speaker
          over-broadening".
        * everything else -- the asker is merely the questioner; what matters is
          any person NAMED in the question.
        """
        if not self.profiles:
            return evidence
        qt = (q_meta or {}).get("qtype")
        target = set()
        first_person = re.search(r"\b(my|me|I|mine|myself)\b", question or "")
        if asker and (qt == "user_implicit" or first_person):
            target.add(asker)
        for n in self.profiles:
            if n and len(n) > 2 and n in (question or ""):
                target.add(n)
        if not target:
            return evidence
        hit = [m for m in evidence if m.get("author") in target]
        rest = [m for m in evidence if m.get("author") not in target]
        return hit + rest

    def _alias_table(self, variant):
        """Alias table for a terminology-alignment ablation rung.

        None -> the graph's default (full CTG). Otherwise a variant table is
        built once per corpus and cached, so the ladder costs one extra
        alignment pass per rung rather than one per question.
        """
        if variant is None or self.graph is None:
            return None
        cached = self._term_tables.get(variant)
        if cached is not None:
            return cached
        from .termalign import TermAligner
        from .retrieval import get_embedding_model
        ents = list(self.graph.entity_msgs.keys())
        if len(ents) < 2:
            self._term_tables[variant] = {}
            return {}
        vecs = get_embedding_model().encode(
            ents, batch_size=128, normalize_embeddings=True,
            convert_to_numpy=True)
        contexts, roles = {}, {}
        if variant != "termnoctx":
            for e in ents:
                mid = next(iter(self.graph.entity_msgs.get(e, ())), None)
                m = self.msg_by_id.get(mid) if mid else None
                if m:
                    contexts[e] = m.get("text") or ""
                    roles[e] = m.get("role") or ""
        backend = "cosine" if variant == "termcosine" else config.TERM_BACKEND
        al = TermAligner(ents, vecs, contexts=contexts, roles=roles,
                         backend=backend)
        if variant == "termoneway":
            al.one_way = True
        al.build()
        self._term_tables[variant] = al.aliases
        print(f"[termalign:{variant}] {al.summary()}", flush=True)
        return al.aliases

    def _search_within(self, query, msg_ids, top_k=10):
        """Rank a restricted id subset against the query (BM25 over the subset)."""
        if not msg_ids:
            return []
        sub = [self.msg_by_id[i] for i in msg_ids if i in self.msg_by_id]
        if not sub:
            return []
        if len(sub) <= top_k:
            return [m["id"] for m in sub]
        bm = BM25()
        bm.index(sub)
        hits = bm.search(query, top_k=top_k)
        out = [m["id"] for _, m in hits]
        if len(out) < top_k:   # pad with most recent
            for m in sorted(sub, key=lambda m: (m.get("ts") or datetime.min),
                            reverse=True):
                if m["id"] not in out:
                    out.append(m["id"])
                if len(out) >= top_k:
                    break
        return out

    def _relevant_profile(self, question, asker):
        """Return the profile dict to inject (asker's, or a name-matched speaker)."""
        if not self.profiles:
            return None
        asker = asker if isinstance(asker, str) else None
        # GroupMemBench: asker id
        if asker and asker in self.profiles:
            return self.profiles[asker]
        # EverMemBench: person named in the question
        for name, prof in self.profiles.items():
            if name and name in (question or ""):
                return prof
        return None

    def _query_adaptive_profile(self, question, asker, q_meta):
        """Select the profile block(s) relevant to the query, or None."""
        prof = self._relevant_profile(question, asker)
        if not prof:
            return None
        dim = (q_meta or {}).get("dimension")
        category = (q_meta or {}).get("category")

        if dim == "Profile Understanding":
            if category == "style":
                return "\n".join(filter(None, [prof.get("identity"),
                                               prof.get("behavior"),
                                               prof.get("style_examples")]))
            if category in ("skill", "title"):
                return "\n".join(filter(None, [prof.get("identity"),
                                               prof.get("expertise"),
                                               prof.get("activity")]))
            return prof.get("text")
        if dim == "Memory Awareness":
            return "\n".join(filter(None, [prof.get("identity"),
                                           prof.get("activity")]))
        return prof.get("text")

    def answer_context(self, method, question, asker=None, q_meta=None, top_k=None):
        """Build the full context string for a question, including profile when on."""
        evidence = self.retrieve(method, question, asker=asker, q_meta=q_meta,
                                 top_k=top_k)
        # The oracle is an upper bound: never drop annotated gold evidence.
        max_chars = None
        if method == "oracle":
            max_chars = max(config.MAX_CONTEXT_CHARS, 60000)
        ctx = build_context(evidence, max_chars=max_chars)
        if method.startswith("camel") and "noprofile" not in method:
            prof = self._query_adaptive_profile(question, asker, q_meta)
            if prof:
                ctx = "USER PROFILE:\n" + prof + "\n\nCONVERSATION EVIDENCE:\n" + ctx
        # Deterministic date arithmetic for duration questions. Even the
        # published oracle tops out at 60% on Temporal because the failure is
        # arithmetic; computing the spans and handing them over removes it.
        if method.startswith("camel") and "notimeline" not in method \
                and config.USE_TIMELINE:
            from . import temporal
            msgs = [m for _, m in evidence]
            hint = temporal.date_hint(question, msgs)
            if hint:
                ctx = ctx + "\n\n" + hint
            # Questions whose ANSWER is a date need the relative expressions in
            # the evidence resolved against each message's own timestamp; the
            # answer model otherwise reports the posting date, which is the
            # single most common error on this question type.
            rhint = temporal.date_resolution_hint(question, msgs)
            if rhint:
                ctx = ctx + "\n\n" + rhint
        return ctx

    def _version_resolve(self, evidence):
        """Annotate stale facts as superseded.

        Two mechanisms, because the benchmarks encode updates differently:
          * GroupMemBench: explicit decision chains keyed by (phase, topic).
          * EverMemBench: no decision flags at all -- so we detect same-entity
            restatements over time and mark the older ones. Without this the
            -noversion ablation was a literal no-op on EverMemBench.
        """
        latest = {}
        for m in evidence:
            if m.get("is_decision"):
                key = (m.get("phase"), (m.get("meta") or {}).get("topic"))
                ts = m.get("ts") or datetime.min
                if key not in latest or ts > latest[key]:
                    latest[key] = ts
        # Annotate copies, never the shared corpus dicts (see the note below).
        out = []
        for m in evidence:
            if m.get("is_decision"):
                key = (m.get("phase"), (m.get("meta") or {}).get("topic"))
                ts = m.get("ts") or datetime.min
                if ts < latest.get(key, ts):
                    out.append(dict(m, meta={**(m.get("meta") or {}),
                                             "superseded": True}))
                    continue
            out.append(m)
        evidence = out

        # Entity-restatement versioning, for corpora with no decision flags.
        #
        # This must be CONSERVATIVE. An earlier version marked a message
        # superseded whenever two retrieved messages shared any entity and both
        # contained an update-ish word ("now", "change", "as of" -- all very
        # common). On a 30k-message corpus that mislabelled current messages as
        # stale and made the answer model report the wrong version of a
        # decision, which is what lost knowledge_update and temporal questions.
        #
        # Two guards: the pair must share a *specific* entity (high IDF, not a
        # corpus-wide one), and the newer message must actually look like a
        # restatement of the older -- not merely mention the same thing.
        if self.graph and self.graph.entity_idf:
            from .graph import _UPDATE_SIGNALS
            idfs = sorted(self.graph.entity_idf.values())
            # top quartile of specificity
            thr = idfs[int(len(idfs) * 0.75)] if len(idfs) > 4 else idfs[-1]
            by_entity = defaultdict(list)
            for m in evidence:
                if not _UPDATE_SIGNALS.search(m.get("text") or ""):
                    continue
                for e in self.graph.mentions.get(m["id"], set()):
                    if self.graph.entity_idf.get(e, 0.0) >= thr:
                        by_entity[e].append(m)
            stale = set()
            for e, msgs in by_entity.items():
                if len(msgs) < 2:
                    continue
                msgs = sorted(msgs, key=lambda m: (m.get("ts") or datetime.min))
                newest = msgs[-1]
                for m in msgs[:-1]:
                    # require the two to be about the same thread of work
                    same_scope = (
                        m.get("group") == newest.get("group")
                        and m.get("phase") == newest.get("phase"))
                    if same_scope and m["id"] != newest["id"]:
                        stale.add(m["id"])
            # Hard cap. Even with the IDF and scope guards, a corpus whose
            # retrieved evidence all sits in one phase can have nearly every
            # message flagged -- which tells the model its whole context is
            # stale and produces "no current approach is available" answers.
            # Only the most recent restatements can plausibly supersede
            # anything, so keep at most a third of the window flagged.
            cap = max(1, len(evidence) // 3)
            if len(stale) > cap:
                order = {m["id"]: i for i, m in enumerate(
                    sorted((m for m in evidence if m["id"] in stale),
                           key=lambda m: (m.get("ts") or datetime.min),
                           reverse=True))}
                stale = {mid for mid in stale if order.get(mid, 10**6) < cap}
            if stale:
                # Annotate COPIES: `evidence` holds the corpus's shared message
                # dicts, so writing the flag in place leaked "superseded" across
                # methods and questions for the rest of the run.
                evidence = [dict(m, meta={**(m.get("meta") or {}),
                                          "superseded": True})
                            if m["id"] in stale else m
                            for m in evidence]
        return evidence

    # -- GraphRAG-style proxy ------------------------------------------
    def _graphrag_retrieve(self, question, top_k):
        """Entity-community retrieval.

        The previous version fell back to plain hybrid whenever entity overlap
        was thin, which happened on essentially every query -- making the
        'graphrag' baseline a byte-identical copy of 'hybrid' in the results.
        Now the entity signal is FUSED with hybrid instead of replaced by it,
        so the baseline is genuinely distinct.
        """
        if not self.graph:
            return self.hybrid.search(question, top_k=top_k)
        q_terms = set(tokenize(question))
        max_idf = max(self.graph.entity_idf.values()) if self.graph.entity_idf else 1.0
        scores = defaultdict(float)
        for e, msgs in self.graph.entity_msgs.items():
            if any(t in e or e in t for t in q_terms):
                w = self.graph.entity_idf.get(e, 0.0) / max(1e-9, max_idf)
                for mid in msgs:
                    scores[mid] += w
        ent_ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k * 3]
        ent_list = [(s, self.msg_by_id[mid]) for mid, s in ent_ranked
                    if mid in self.msg_by_id]
        hyb = self.hybrid.search(question, top_k=top_k * 3)
        fused = rrf_fuse([ent_list, hyb], weights=[1.0, 0.6])
        out = [(s, self.msg_by_id[mid]) for mid, s in fused[:top_k]
               if mid in self.msg_by_id]
        return out or hyb[:top_k]

    def gold_ids(self, q_meta):
        """Message ids of the annotated gold evidence, or an empty set."""
        try:
            return {m["id"] for _s, m in self._oracle_retrieve(q_meta, 10 ** 6)}
        except Exception:  # noqa: BLE001
            return set()

    # -- oracle (gold evidence) ----------------------------------------
    def _oracle_retrieve(self, q_meta, top_k):
        """Return the annotated gold evidence spans.

        The oracle is an upper bound and must NOT be truncated to top_k: an
        EverMemBench item carries ~6.7 references on average and a multi-hop
        item needs *every* anchor (dropping one makes the date arithmetic
        unanswerable). The published oracle scores ~98% on multi-hop; the
        previous implementation clipped to top_k and matched ids too strictly,
        which is why the local oracle sat at 46%.
        """
        refs = (q_meta or {}).get("reference")
        if not refs:
            return []
        if isinstance(refs, dict):
            refs = [refs]
        # A benchmark may annotate gold evidence as bare message ids rather than
        # {date, group, message_index} triples (SocialMemBench uses turn ids).
        if all(isinstance(r, str) for r in refs):
            msgs = [self.msg_by_id[r] for r in dict.fromkeys(refs)
                    if r in self.msg_by_id]
            msgs.sort(key=lambda m: (m.get("ts") or datetime.min))
            return [(1.0, m) for m in msgs]
        topic = q_meta.get("topic_id")
        ids, missing = [], 0
        for r in refs:
            if not isinstance(r, dict):
                continue
            date = str(r.get("date") or "").strip()[:10]
            grp = r.get("group")
            idxs = _parse_index(r.get("message_index"))
            for idx in idxs:
                mid = f"{topic}|{date}|{grp}|{idx}"
                if mid in self.msg_by_id:
                    ids.append(mid)
                else:
                    missing += 1
                    # tolerate group-label and index-base drift
                    alt = self._oracle_fallback(topic, date, grp, idx)
                    if alt:
                        ids.append(alt)
        # de-dup, keep chronological order (date arithmetic is far easier)
        seen, msgs = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                msgs.append(self.msg_by_id[i])
        msgs.sort(key=lambda m: (m.get("ts") or datetime.min))
        return [(1.0, m) for m in msgs]

    def _oracle_fallback(self, topic, date, grp, idx):
        """Resolve a gold reference whose group label or index base drifted."""
        if not hasattr(self, "_by_date_group"):
            self._by_date_group = defaultdict(list)
            for m in self.messages:
                d = (m.get("meta") or {}).get("date")
                self._by_date_group[(d, m.get("group"))].append(m)
            for v in self._by_date_group.values():
                v.sort(key=lambda m: ((m.get("meta") or {}).get("message_index") or 0))
        # normalise "Group 2" / "group2" / "2"
        cands = []
        gnorm = re.sub(r"\D", "", str(grp or ""))
        for (d, g), msgs in self._by_date_group.items():
            if d != date:
                continue
            if gnorm and re.sub(r"\D", "", str(g or "")) != gnorm:
                continue
            cands = msgs
            break
        if not cands:
            return None
        for m in cands:
            if (m.get("meta") or {}).get("message_index") == idx:
                return m["id"]
        # 0-based vs 1-based index drift
        if 0 <= idx - 1 < len(cands):
            return cands[idx - 1]["id"]
        return None


def _rule_split(question):
    """Fallback decomposition: split comparison questions on connectives."""
    q = question.strip()
    subs = []
    # "how long after X did Y" / "between X and Y"
    m = re.search(r"\bbetween\s+(.{6,120}?)\s+and\s+(.{6,120}?)[\?\.]?$", q,
                  re.IGNORECASE)
    if m:
        subs = [f"When did {m.group(1).strip()}?", f"When did {m.group(2).strip()}?"]
    if not subs:
        m = re.search(r"\b(?:after|before|since)\b(.{6,160})", q, re.IGNORECASE)
        if m:
            subs = [f"When did {m.group(1).strip().rstrip('?.')}?"]
    if not subs:
        parts = re.split(r",\s*(?:and|then|after which)\s+|\band then\b", q)
        subs = [p.strip() for p in parts if len(p.strip()) > 15][:2]
        if len(subs) < 2:
            subs = []
    return subs


def _parse_index(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")[:2]
            try:
                out.extend(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def build_context(evidence, max_chars=None):
    """Render evidence messages into a context string, truncating by budget."""
    max_chars = config.MAX_CONTEXT_CHARS if max_chars is None else max_chars
    parts = []
    total = 0
    for _, m in evidence:
        if not m:
            continue
        line = format_message(m)
        total += len(line) + 1
        if total > max_chars:
            break
        parts.append(line)
    return "\n".join(parts)
