"""Published memory / retrieval baselines, reimplemented over a shared corpus.

Every baseline here follows the algorithm of a peer-reviewed or widely-adopted
system, but runs against the SAME corpus, embedding model, answer LLM and judge
as CAMEL, so the comparison isolates memory structure rather than backbone.
Running the vendors' hosted APIs would confound the comparison (different answer
models) and is not reproducible offline on the cluster.

  mem0      Mem0: LLM-extracted atomic facts + ADD/UPDATE/DELETE consolidation
            (Chhikara et al., 2025). Distinct from `flat_mem`, which skips
            consolidation entirely.
  zep       Zep / Graphiti: temporal knowledge graph with validity intervals and
            fact invalidation (Rasmussen et al., 2025).
  hipporag  HippoRAG: personalised PageRank over an OpenIE graph, seeded by
            query-linked entities (Gutiérrez et al., NeurIPS 2024).
  raptor    RAPTOR: recursive clustering + summarisation into a tree, collapsed
            -tree retrieval (Sarthi et al., ICLR 2024).
  readagent ReadAgent: gist memory over episode pages + look-up of pages whose
            gist matches the query (Lee et al., ICML 2024).
  memgpt    MemGPT / Letta: paged main-context + archival store with recall
            (Packer et al., 2023).

Artefacts (facts, summaries, gists, graphs) are cached under CACHE_DIR so a
sweep pays the LLM ingestion cost once per corpus.
"""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from . import config
from .retrieval import BM25, get_embedding_model, rrf_fuse, tokenize


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _model_tag():
    """Filesystem-safe tag for the model that builds ingestion artefacts.

    Mem0, Zep, RAPTOR and ReadAgent all call an LLM at ingestion time, so their
    cached facts/summaries/gists belong to the model that produced them. Keying
    the cache on the corpus alone would let a backbone-transfer run silently
    reuse another model's artefacts and report it as that backbone's result.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", config.ANSWER_MODEL) or "default"


def _cache_dir(name):
    d = config.CACHE_DIR / "baselines" / name / _model_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _embed(texts, batch_size=None):
    model = get_embedding_model()
    return model.encode(
        texts, batch_size=batch_size or config.EMBED_BATCH_SIZE,
        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def _llm_map(prompts, system, max_tokens=600, workers=16):
    """Run one LLM call per prompt, in parallel, tolerating failures."""
    from .llm import LLMClient

    def one(p):
        try:
            return LLMClient(max_tokens=max_tokens).complete(system, p)
        except Exception:  # noqa: BLE001
            return ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, prompts))


def _fmt(m):
    from .engine import format_message
    return format_message(m)


def _top(sims, k):
    k = min(k, len(sims))
    if k <= 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])]


# ---------------------------------------------------------------------------
# Mem0 -- extracted facts with consolidation
# ---------------------------------------------------------------------------
_MEM0_SYS = (
    "You maintain a memory store for a workplace group chat. From the "
    "conversation window, output the atomic facts worth remembering, one JSON "
    "object per line, no preamble:\n"
    '{"fact": "<self-contained statement incl. who/what/when>", '
    '"entity": "<the main project, artefact or person the fact is about>"}\n'
    "Only durable facts (decisions, assignments, deadlines, deliverables, "
    "status changes). Skip pleasantries and small talk."
)


class Mem0Memory:
    """Fact store with ADD / UPDATE consolidation keyed on entity."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.facts = []          # (text, src_id, ts)
        self.emb = None

    def build(self, window=20):
        key = self.corpus.corpus_key or "anon"
        cd = _cache_dir("mem0")
        fp, ep = cd / f"{key}.json", cd / f"{key}.npy"
        if fp.exists() and ep.exists():
            facts = [tuple(x) for x in json.loads(fp.read_text())]
            emb = np.load(ep)
            # Guard against a stale cache: if the matrix and the fact list have
            # drifted apart, the dot product later fails with an opaque
            # broadcast error, so rebuild instead.
            if len(emb) == len(facts):
                self.facts, self.emb = facts, emb
                return self
            print(f"[{fp.stem}] cache mismatch "
                  f"({len(emb)} vectors vs {len(facts)} facts); rebuilding")

        msgs = self.corpus.messages
        chunks, srcs = [], []
        for i in range(0, len(msgs), window):
            ch = msgs[i:i + window]
            chunks.append("\n".join(_fmt(m) for m in ch))
            srcs.append(ch[-1]["id"])
        raw = _llm_map(chunks, _MEM0_SYS)

        # parse
        parsed = []   # (fact, entity, src, ts)
        for text, src in zip(raw, srcs):
            ts = self.corpus.msg_by_id[src].get("ts") if src in self.corpus.msg_by_id else None
            for line in (text or "").splitlines():
                line = line.strip().strip(",")
                if not line:
                    continue
                fact = ent = None
                if line.startswith("{"):
                    try:
                        o = json.loads(line)
                        fact, ent = o.get("fact"), (o.get("entity") or "")
                    except Exception:  # noqa: BLE001
                        pass
                if not fact:
                    fact = line.strip(" -•*")
                    ent = ""
                if fact and len(fact) > 12:
                    parsed.append((fact, (ent or "").lower().strip(), src, ts))

        # consolidation: within an entity, a later fact that is highly similar
        # to an earlier one UPDATES it (the older is dropped) -- this is the
        # step that distinguishes Mem0 from a flat append-only fact list.
        parsed.sort(key=lambda x: (x[3] or datetime.min))
        by_ent = defaultdict(list)
        for f in parsed:
            by_ent[f[1]].append(f)
        kept = []
        for ent, items in by_ent.items():
            if not ent or len(items) < 2:
                kept.extend(items)
                continue
            vecs = _embed([i[0] for i in items])
            drop = set()
            for a in range(len(items)):
                if a in drop:
                    continue
                for b in range(a + 1, len(items)):
                    if b in drop:
                        continue
                    if float(vecs[a] @ vecs[b]) >= 0.90:
                        drop.add(a)      # older superseded by newer
                        break
            kept.extend(i for n, i in enumerate(items) if n not in drop)

        self.facts = [(f, s, (t.isoformat() if t else None)) for f, _e, s, t in kept]
        self.emb = _embed([f for f, _, _ in self.facts]) if self.facts else np.zeros((0, 1024), np.float32)
        fp.write_text(json.dumps(self.facts, ensure_ascii=False))
        np.save(ep, self.emb)
        return self

    def search(self, query, top_k):
        if not self.facts:
            return []
        qv = _embed([query])[0]
        sims = self.emb @ qv
        out = []
        for i in _top(sims, top_k):
            _f, src, _t = self.facts[i]
            m = self.corpus.msg_by_id.get(src)
            if m is not None:
                out.append((float(sims[i]), m))
        return out


# ---------------------------------------------------------------------------
# Zep / Graphiti -- temporal knowledge graph with fact invalidation
# ---------------------------------------------------------------------------
_ZEP_SYS = (
    "Extract temporal knowledge-graph edges from this workplace conversation "
    "window. One JSON object per line, no preamble:\n"
    '{"subject": "...", "predicate": "...", "object": "...", '
    '"valid_from": "YYYY-MM-DD or null"}\n'
    "Capture assignments, decisions, deadlines and status changes only."
)


class ZepMemory:
    """Bi-temporal edge store: newer edges invalidate same (subject,predicate)."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.edges = []   # (text, src_id, ts_iso, invalid)
        self.emb = None

    def build(self, window=20):
        key = self.corpus.corpus_key or "anon"
        cd = _cache_dir("zep")
        fp, ep = cd / f"{key}.json", cd / f"{key}.npy"
        if fp.exists() and ep.exists():
            edges = [tuple(x) for x in json.loads(fp.read_text())]
            emb = np.load(ep)
            if len(emb) == len(edges):
                self.edges, self.emb = edges, emb
                return self
            print(f"[{fp.stem}] cache mismatch; rebuilding")

        msgs = self.corpus.messages
        chunks, srcs = [], []
        for i in range(0, len(msgs), window):
            ch = msgs[i:i + window]
            chunks.append("\n".join(_fmt(m) for m in ch))
            srcs.append(ch[-1]["id"])
        raw = _llm_map(chunks, _ZEP_SYS)

        triples = []
        for text, src in zip(raw, srcs):
            ts = self.corpus.msg_by_id[src].get("ts") if src in self.corpus.msg_by_id else None
            for line in (text or "").splitlines():
                line = line.strip().strip(",")
                if not line.startswith("{"):
                    continue
                try:
                    o = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                s, p, ob = (o.get("subject") or "", o.get("predicate") or "",
                            o.get("object") or "")
                if not (s and p):
                    continue
                triples.append(((s, p), f"{s} {p} {ob}", src, ts))

        # invalidation: for one (subject,predicate) only the newest edge stays
        triples.sort(key=lambda x: (x[3] or datetime.min))
        newest = {}
        for k, text, src, ts in triples:
            newest[k] = (text, src, ts)
        edges = []
        for k, text, src, ts in triples:
            live = newest[k][1] == src and newest[k][0] == text
            label = text if live else text + " [INVALIDATED by a later update]"
            edges.append((label, src, ts.isoformat() if ts else None, not live))
        self.edges = edges
        self.emb = _embed([e[0] for e in edges]) if edges else np.zeros((0, 1024), np.float32)
        fp.write_text(json.dumps(self.edges, ensure_ascii=False))
        np.save(ep, self.emb)
        return self

    def search(self, query, top_k):
        if not self.edges:
            return []
        qv = _embed([query])[0]
        sims = self.emb @ qv
        # prefer live edges
        adj = sims - 0.05 * np.array([e[3] for e in self.edges], dtype=np.float32)
        out = []
        for i in _top(adj, top_k):
            m = self.corpus.msg_by_id.get(self.edges[i][1])
            if m is not None:
                out.append((float(sims[i]), m))
        return out


# ---------------------------------------------------------------------------
# HippoRAG -- personalised PageRank over an entity graph
# ---------------------------------------------------------------------------
class HippoRAGMemory:
    """Query entities seed a PPR walk over the message-entity bipartite graph."""

    def __init__(self, corpus):
        self.corpus = corpus

    def build(self):
        return self

    def search(self, query, top_k, alpha=0.5, iters=12):
        g = self.corpus.graph
        if g is None:
            return self.corpus.hybrid.search(query, top_k=top_k)
        q_terms = set(tokenize(query))
        # seed on entities the query mentions, weighted by IDF
        seeds = {}
        for e in g.entity_msgs:
            if any(t in e or e in t for t in q_terms):
                seeds[e] = g.entity_idf.get(e, 1.0)
        if not seeds:
            return self.corpus.hybrid.search(query, top_k=top_k)
        tot = sum(seeds.values()) or 1.0
        ent_p = {e: w / tot for e, w in seeds.items()}
        reset = dict(ent_p)

        msg_p = defaultdict(float)
        for _ in range(iters):
            # entity -> message
            msg_p = defaultdict(float)
            for e, p in ent_p.items():
                targets = g.entity_msgs.get(e, ())
                if not targets:
                    continue
                share = p / len(targets)
                for mid in targets:
                    msg_p[mid] += share
            # message -> entity
            new_ent = defaultdict(float)
            for mid, p in msg_p.items():
                ents = g.mentions.get(mid, ())
                if not ents:
                    continue
                share = p / len(ents)
                for e in ents:
                    new_ent[e] += share
            ent_p = {e: (1 - alpha) * v for e, v in new_ent.items()}
            for e, v in reset.items():
                ent_p[e] = ent_p.get(e, 0.0) + alpha * v
        ranked = sorted(msg_p.items(), key=lambda x: -x[1])[:top_k]
        out = [(s, self.corpus.msg_by_id[mid]) for mid, s in ranked
               if mid in self.corpus.msg_by_id]
        return out or self.corpus.hybrid.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# RAPTOR -- recursive clustering + summarisation tree
# ---------------------------------------------------------------------------
_RAPTOR_SYS = ("Summarise this group-chat excerpt in 3-4 sentences. Preserve "
               "names, dates, decisions and deliverables. No preamble.")


class RaptorMemory:
    """Collapsed-tree retrieval over leaf chunks + recursive summaries."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.nodes = []   # (text, src_id, level)
        self.emb = None

    def build(self, chunk=12, levels=2, branch=8):
        key = self.corpus.corpus_key or "anon"
        cd = _cache_dir("raptor")
        fp, ep = cd / f"{key}.json", cd / f"{key}.npy"
        if fp.exists() and ep.exists():
            nodes = [tuple(x) for x in json.loads(fp.read_text())]
            emb = np.load(ep)
            if len(emb) == len(nodes):
                self.nodes, self.emb = nodes, emb
                return self
            print(f"[{fp.stem}] cache mismatch; rebuilding")

        msgs = self.corpus.messages
        nodes, texts = [], []
        for i in range(0, len(msgs), chunk):
            ch = msgs[i:i + chunk]
            t = "\n".join(_fmt(m) for m in ch)
            nodes.append((t, ch[len(ch) // 2]["id"], 0))
            texts.append(t)

        cur = list(range(len(nodes)))
        for lvl in range(1, levels + 1):
            if len(cur) <= 1:
                break
            groups = [cur[i:i + branch] for i in range(0, len(cur), branch)]
            prompts = ["\n\n".join(nodes[i][0] for i in g)[:12000] for g in groups]
            summaries = _llm_map(prompts, _RAPTOR_SYS, max_tokens=300)
            new = []
            for g, s in zip(groups, summaries):
                if not s.strip():
                    continue
                nodes.append((s.strip(), nodes[g[len(g) // 2]][1], lvl))
                new.append(len(nodes) - 1)
            cur = new

        self.nodes = nodes
        self.emb = _embed([n[0] for n in nodes])
        fp.write_text(json.dumps(self.nodes, ensure_ascii=False))
        np.save(ep, self.emb)
        return self

    def search(self, query, top_k):
        if not self.nodes:
            return []
        qv = _embed([query])[0]
        sims = self.emb @ qv
        out, seen = [], set()
        for i in _top(sims, top_k * 3):
            src = self.nodes[i][1]
            if src in seen:
                continue
            seen.add(src)
            m = self.corpus.msg_by_id.get(src)
            if m is not None:
                out.append((float(sims[i]), m))
            if len(out) >= top_k:
                break
        return out


# ---------------------------------------------------------------------------
# ReadAgent -- gist memory + page look-up
# ---------------------------------------------------------------------------
_GIST_SYS = ("Compress this group-chat page into a short gist that preserves "
             "who did what, dates and decisions. 2-3 sentences, no preamble.")


class ReadAgentMemory:
    """Episode pages compressed to gists; retrieval looks up matching pages."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.pages = []   # (gist, [msg ids])
        self.emb = None

    def build(self, page=25):
        key = self.corpus.corpus_key or "anon"
        cd = _cache_dir("readagent")
        fp, ep = cd / f"{key}.json", cd / f"{key}.npy"
        if fp.exists() and ep.exists():
            pages = [tuple(x) for x in json.loads(fp.read_text())]
            emb = np.load(ep)
            if len(emb) == len(pages):
                self.pages, self.emb = pages, emb
                return self
            print(f"[{fp.stem}] cache mismatch; rebuilding")
        msgs = self.corpus.messages
        chunks, ids = [], []
        for i in range(0, len(msgs), page):
            ch = msgs[i:i + page]
            chunks.append("\n".join(_fmt(m) for m in ch))
            ids.append([m["id"] for m in ch])
        gists = _llm_map(chunks, _GIST_SYS, max_tokens=200)
        self.pages = [(g.strip(), mid) for g, mid in zip(gists, ids) if g.strip()]
        self.emb = _embed([p[0] for p in self.pages]) if self.pages else \
            np.zeros((0, 1024), np.float32)
        fp.write_text(json.dumps(self.pages, ensure_ascii=False))
        np.save(ep, self.emb)
        return self

    def search(self, query, top_k):
        if not self.pages:
            return []
        qv = _embed([query])[0]
        sims = self.emb @ qv
        # look up the best pages, then rank messages inside them
        cand = []
        for i in _top(sims, 4):
            cand.extend(self.pages[i][1])
        sub = [self.corpus.msg_by_id[m] for m in cand if m in self.corpus.msg_by_id]
        if not sub:
            return []
        bm = BM25()
        bm.index(sub)
        hits = bm.search(query, top_k=top_k)
        if len(hits) < top_k:
            have = {m["id"] for _, m in hits}
            hits += [(0.0, m) for m in sub if m["id"] not in have][:top_k - len(hits)]
        return hits[:top_k]


# ---------------------------------------------------------------------------
# MemGPT / Letta -- paged main context + archival recall
# ---------------------------------------------------------------------------
class MemGPTMemory:
    """Recency-weighted main context plus embedding recall over the archive."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.dense = None

    def build(self):
        self.dense = self.corpus.hybrid.dense
        return self

    def search(self, query, top_k, main_frac=0.3):
        n_main = max(1, int(top_k * main_frac))
        n_arch = top_k - n_main
        # "main context": the most recent messages, always resident
        recent = sorted(self.corpus.messages,
                        key=lambda m: (m.get("ts") or datetime.min))[-n_main:]
        arch = self.dense.search(query, top_k=n_arch * 2)
        out, seen = [], set()
        for s, m in arch:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append((s, m))
            if len(out) >= n_arch:
                break
        for m in recent:
            if m["id"] not in seen:
                out.append((0.0, m))
        return out[:top_k]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_BUILDERS = {
    "mem0": Mem0Memory,
    "zep": ZepMemory,
    "hipporag": HippoRAGMemory,
    "raptor": RaptorMemory,
    "readagent": ReadAgentMemory,
    "memgpt": MemGPTMemory,
}

# baselines whose ingestion needs LLM calls (so the driver can warn about cost)
LLM_INGEST = {"mem0", "zep", "raptor", "readagent"}


def get(corpus, name):
    if name not in corpus._ext:
        corpus._ext[name] = _BUILDERS[name](corpus).build()
    return corpus._ext[name]


def retrieve(corpus, name, question, asker, q_meta, top_k):
    return get(corpus, name).search(question, top_k)
