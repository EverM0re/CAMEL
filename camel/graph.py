"""Graph construction for CAMEL.

Builds a heterogeneous memory graph over a message corpus and provides
multi-hop expansion for retrieval.  The graph encodes:

  - reply/thread structure (REPLY_TO)
  - authorship / groups / phases
  - entity mentions (extracted terms)
  - cross-role term alignment (SAME_AS via embedding near-synonymy)
  - temporal version chains (UPDATES)
  - event anchors (timestamped "X was completed/started" statements)

Everything is LLM-free at ingestion.
"""
import math
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from . import config

# Patterns for entity extraction: capitalized multi-word phrases, quoted strings,
# and specific artifacts (URLs, versions, ids).
# Capitalised phrases must not run across a sentence boundary: the trailing
# "." of an abbreviation is allowed only when a letter/digit follows it, so
# "Carbon Platform. Coordinating" yields "Carbon Platform", not one fused term.
_ENTITY_RE = re.compile(
    r'(?:"[^"]{2,40}")'
    r"|\b[A-Z][A-Za-z0-9&\-]*(?:\.[A-Za-z0-9])?"
    r"(?:\s+[A-Z][A-Za-z0-9&\-]*(?:\.[A-Za-z0-9])?){0,5}\b"
    r"|\b[A-Z]{2,8}\b"
    r"|\b[A-Za-z]+-\d+\b"
    r"|\bv?\d+\.\d+(?:\.\d+)?\b"
)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on", "with",
    "we", "our", "i", "you", "it", "this", "that", "is", "are", "be", "will",
    "hello", "hi", "good", "morning", "thanks", "thank", "please", "team", "everyone",
}
_UPDATE_SIGNALS = re.compile(
    r"\b(update|updated|revised|revision|changed|change|supersede|supersedes|"
    r"as of|now|no longer|instead|rather than|moving to|postpone|postponed|"
    r"reschedul\w*|extend\w*|defer|bring forward|overrid\w*)\b",
    re.IGNORECASE,
)
# Completion / start events: the anchors multi-hop "how many days between"
# questions must land on.
_EVENT_SIGNALS = re.compile(
    r"\b(complet\w*|finish\w*|finaliz\w*|deliver\w*|submit\w*|"
    r"start\w*|begin|began|begun|kick\w*off|launch\w*|"
    r"approv\w*|sign\w*off|releas\w*|deploy\w*|merg\w*|"
    r"upload\w*|publish\w*|handover|hand over|wrapp\w*\s+up|done)\b",
    re.IGNORECASE,
)


def _norm_term(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


class MemoryGraph:
    def __init__(self):
        self.messages = []
        self.msg_by_id = {}
        self.children = defaultdict(list)      # parent_id -> [child ids]
        self.parent = {}                        # msg_id -> parent id
        self.author_msgs = defaultdict(list)    # author -> [msg ids]
        self.group_msgs = defaultdict(list)     # group -> [msg ids]
        self.phase_msgs = defaultdict(list)     # phase -> [msg ids]
        self.mentions = defaultdict(set)        # msg_id -> set of entities
        self.entity_msgs = defaultdict(set)     # entity -> set of msg ids
        self.entity_idf = {}                    # entity -> idf (informativeness)
        self.entity_aliases = defaultdict(set)  # entity -> {aliases}
        self.term_relations = {}                # (a,b) -> typed relation
        self.term_stats = ""
        self.version_of = {}                    # msg_id -> msg_id it updates
        self.decision_chain = defaultdict(list) # (phase,topic) -> [msg ids] chronological
        self.event_msgs = []                    # msg ids that state an event
        self.date_msgs = defaultdict(list)      # 'YYYY-MM-DD' -> [msg ids]

    # ------------------------------------------------------------------
    def build(self, messages, embeddings=None, alias_sim=None, use_alias=True,
              max_entities=3000):
        alias_sim = config.ALIAS_SIM if alias_sim is None else alias_sim
        self.messages = messages
        self.msg_by_id = {m["id"]: m for m in messages}

        # thread structure
        for m in messages:
            if m.get("reply_to") and m["reply_to"] in self.msg_by_id:
                self.parent[m["id"]] = m["reply_to"]
                self.children[m["reply_to"]].append(m["id"])
            self.author_msgs[m.get("author")].append(m["id"])
            self.group_msgs[m.get("group")].append(m["id"])
            if m.get("phase"):
                self.phase_msgs[m["phase"]].append(m["id"])
            ts = m.get("ts")
            if ts:
                self.date_msgs[ts.strftime("%Y-%m-%d")].append(m["id"])
            if _EVENT_SIGNALS.search(m.get("text") or ""):
                self.event_msgs.append(m["id"])

        # entity extraction + mentions (raw)
        raw_mentions = defaultdict(set)
        raw_entity_msgs = defaultdict(set)
        for m in messages:
            ents = self._extract_entities(m["text"])
            for e in ents:
                raw_mentions[m["id"]].add(e)
                raw_entity_msgs[e].add(m["id"])

        # Prune to a tractable, informative entity set: drop singletons and
        # ultra-generic terms, then keep the most INFORMATIVE survivors.
        #
        # The cap is a quantile of the observed frequencies, not a fixed
        # fraction of the corpus. A fixed cap is fragile: on a corpus where
        # every salient project name is mentioned in >5% of messages it
        # discarded the entire entity set, silently turning graph expansion
        # into a no-op. A quantile always keeps a usable set.
        freqs = {e: len(v) for e, v in raw_entity_msgs.items()}
        multi = sorted(f for f in freqs.values() if f >= 2)
        if multi:
            # drop the top 2% most frequent (the "the/status/working" tier)
            cap = multi[int(len(multi) * 0.98)] if len(multi) > 50 else multi[-1]
            cap = max(cap, 2)
        else:
            cap = 1
        candidates = [e for e, f in freqs.items() if 2 <= f <= cap]
        if not candidates:   # degenerate corpus: fall back to all repeated terms
            candidates = [e for e, f in freqs.items() if f >= 2]
        n_msgs_ = max(1, len(messages))
        # rank by idf-weighted frequency: specific terms that still recur
        candidates.sort(
            key=lambda e: -(freqs[e] * math.log(1 + n_msgs_ / max(1, freqs[e]))))
        keep = set(candidates[:max_entities])

        for m in messages:
            for e in raw_mentions[m["id"]]:
                if e in keep:
                    self.mentions[m["id"]].add(e)
                    self.entity_msgs[e].add(m["id"])

        # informativeness (idf) of each kept entity: higher = more specific
        n_msgs = max(1, len(messages))
        self.entity_idf = {
            e: math.log(1 + n_msgs / max(1, len(v)))
            for e, v in self.entity_msgs.items()
        }

        # decision chains (GroupMemBench versioning)
        for m in messages:
            if m.get("is_decision"):
                key = (m.get("phase"), (m.get("meta") or {}).get("topic"))
                self.decision_chain[key].append(m["id"])
        for key in self.decision_chain:
            self.decision_chain[key].sort(
                key=lambda mid: (self.msg_by_id[mid]["ts"] or datetime.min))

        # explicit version links from decision_change metadata
        for m in messages:
            ch = (m.get("meta") or {}).get("decision_change")
            if ch and isinstance(ch, dict):
                prior = ch.get("previous") or ch.get("prior") or ch.get("from")
                if prior:
                    self.version_of[m["id"]] = str(prior)

        # cross-role term alignment via embedding near-synonymy
        if use_alias and self.entity_msgs:
            self._build_aliases(alias_sim)
        return self

    def _extract_entities(self, text):
        out = set()
        for m in _ENTITY_RE.finditer(text or ""):
            term = _norm_term(m.group(0))
            if not term or term in _STOPWORDS or len(term) < 3:
                continue
            if any(c.isdigit() for c in term) and len(term) < 5:
                continue
            out.add(term)
        return out

    def _build_aliases(self, sim):
        """Contextual Term Grounding: candidate retrieval + relation inference.

        Similarity is used only to generate candidates; whether two terms are
        aliases is decided by bidirectional entailment (see termalign.py). Only
        `equivalent` pairs become alias edges, so broader/narrower relations
        ("Finance" vs "Finance Ops") no longer expand into wrong evidence.
        """
        ents = list(self.entity_msgs.keys())
        if len(ents) < 2:
            return
        from .retrieval import get_embedding_model
        model = get_embedding_model()
        vecs = model.encode(ents, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)

        # A representative utterance + speaker role per term, so the relation
        # model can condition on the context the term was actually used in.
        contexts, roles = {}, {}
        for e in ents:
            mid = next(iter(self.entity_msgs.get(e, ())), None)
            m = self.msg_by_id.get(mid) if mid else None
            if m:
                contexts[e] = m.get("text") or ""
                roles[e] = m.get("role") or ""

        from .termalign import TermAligner
        al = TermAligner(ents, vecs, contexts=contexts, roles=roles,
                         backend=config.TERM_BACKEND, tau=config.TERM_TAU)
        al.build()
        self.term_relations = al.relations
        self.term_stats = al.summary()
        for a, bs in al.aliases.items():
            for b in bs:
                self.entity_aliases[a].add(b)
        print(f"[termalign] {al.summary()}", flush=True)

    # ------------------------------------------------------------------
    def expand(self, seed_msg_ids, hops=None, budget=60, query_terms=None):
        """Scored multi-hop expansion of seed messages along graph edges.

        Returns a list of (score, msg_id) for NON-SEED neighbours, ordered by
        descending edge-evidence score.

        The previous implementation returned an unordered ``set``, so graph
        neighbours entered the answer context in arbitrary hash order and
        displaced strongly-ranked hybrid hits. Every neighbour is now scored by
        (a) the type of edge that reached it, (b) the IDF of the shared entity,
        and (c) a decay per hop, so only the best-supported neighbours survive.
        """
        hops = config.GRAPH_HOPS if hops is None else hops
        seeds = list(seed_msg_ids)
        seen = set(seeds)
        scores = defaultdict(float)
        query_terms = query_terms or set()

        # only expand along entities more specific than the median
        idf_vals = sorted(self.entity_idf.values()) if self.entity_idf else []
        idf_thr = idf_vals[len(idf_vals) // 2] if idf_vals else 1e9
        max_idf = max(idf_vals) if idf_vals else 1.0

        # Seeds are rank-weighted and the mass is normalised, so widening the
        # anchor set improves reachability without inflating the scores of
        # neighbours that many anchors happen to share.
        frontier = {mid: 1.0 / (1.0 + 0.05 * i) for i, mid in enumerate(seeds)}
        best_edge = defaultdict(float)   # neighbour -> strongest single edge
        for hop in range(hops):
            decay = 0.6 ** hop
            new = defaultdict(float)
            for mid, w in frontier.items():
                m = self.msg_by_id.get(mid)
                if not m:
                    continue
                # thread: parent + children (explicit reply structure) -- the
                # highest-confidence edge type.
                if mid in self.parent:
                    p_ = self.parent[mid]
                    val = 1.0 * w * decay
                    new[p_] += val
                    best_edge[p_] = max(best_edge[p_], val)
                for c in self.children.get(mid, [])[:5]:
                    val = 0.95 * w * decay
                    new[c] += val
                    best_edge[c] = max(best_edge[c], val)
                # decision chain: same (phase, topic) -> version evolution
                if m.get("is_decision"):
                    key = (m.get("phase"), (m.get("meta") or {}).get("topic"))
                    chain = self.decision_chain.get(key, [])
                    if mid in chain:
                        i = chain.index(mid)
                        for nb in chain[max(0, i - 3):i + 4]:
                            val = 0.85 * w * decay
                            new[nb] += val
                            best_edge[nb] = max(best_edge[nb], val)
                # Entity edges are the WEAKEST evidence type and the most
                # numerous, so they are (a) capped well below the structural
                # edges and (b) divided by their fan-out. Without the fan-out
                # penalty a generic entity shared by 12 near-duplicates
                # out-scored an explicit reply edge by ~45x, burying the real
                # second hop underneath its own distractors.
                for e in self.mentions.get(mid, set()):
                    idf = self.entity_idf.get(e, 0.0)
                    if idf < idf_thr:
                        continue
                    boost = 1.5 if (query_terms and any(
                        t in e or e in t for t in query_terms)) else 1.0
                    targets = list(self.entity_msgs.get(e, set()))
                    if len(targets) > 12:
                        targets = targets[:12]
                    if not targets:
                        continue
                    ew = (0.35 * (idf / max(1e-9, max_idf)) * boost * w * decay
                          / len(targets))
                    for nb in targets:
                        new[nb] += ew
                        best_edge[nb] = max(best_edge[nb], ew)
            frontier = {k: v for k, v in new.items() if k not in seen}
            if not frontier:
                break
            for k, v in frontier.items():
                # A neighbour is ranked by its STRONGEST single edge, not by the
                # sum over many weak ones, with the accumulated mass acting only
                # as a small tie-break. Summing let a message reachable from
                # dozens of anchors through one generic entity outrank a message
                # sitting on an explicit reply edge -- exactly the second hop the
                # expansion exists to find.
                scores[k] = max(scores[k],
                                best_edge.get(k, 0.0) + 0.001 * min(v, 1.0))
            seen.update(frontier)
            # keep the frontier tractable
            if len(frontier) > 400:
                frontier = dict(sorted(frontier.items(), key=lambda x: -x[1])[:400])

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(s, mid) for mid, s in ranked[:budget]]

    def alias_expand(self, query, budget=40, aliases=None):
        """Message ids whose entities are cross-role aliases of query terms.

        Scored by alias strength * entity IDF so the caller can merge these into
        a single ranked list rather than appending them blindly.
        """
        from .retrieval import tokenize
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        max_idf = max(self.entity_idf.values()) if self.entity_idf else 1.0
        scores = defaultdict(float)
        table = self.entity_aliases if aliases is None else aliases
        for e, alist in table.items():
            if not any(t in e or e in t for t in q_terms):
                continue
            for a in alist:
                idf = self.entity_idf.get(a, 0.0)
                for mid in list(self.entity_msgs.get(a, set()))[:8]:
                    scores[mid] = max(scores[mid], idf / max(1e-9, max_idf))
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:budget]
        return [(s, mid) for mid, s in ranked]

    def neighbours_by_date(self, date_str, window=0, budget=40):
        """Message ids on a given date (optionally +/- `window` days)."""
        from datetime import timedelta
        try:
            d0 = datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return []
        out = []
        for off in range(-window, window + 1):
            key = (d0 + timedelta(days=off)).strftime("%Y-%m-%d")
            out.extend(self.date_msgs.get(key, []))
        return out[:budget]

    def get_versions(self, msg_id):
        """Return messages in the same decision/topic chain, chronological."""
        m = self.msg_by_id.get(msg_id)
        if not m or not m["is_decision"]:
            return []
        key = (m.get("phase"), (m.get("meta") or {}).get("topic"))
        chain = self.decision_chain.get(key, [])
        return [self.msg_by_id[i] for i in chain if i in self.msg_by_id]
