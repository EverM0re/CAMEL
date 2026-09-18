"""Retrieval primitives: BM25 (lexical), bge-m3 dense, hybrid fusion, reranking.

All retrievers operate over a list of message dicts (see data.py) and expose:

    index(messages)            -> build the index
    search(query, top_k)       -> list of (score, message)

Dense embeddings are computed once and cached to disk (numpy) so multiple
baselines / ablations reuse the same vectors.
"""
import math
import re
import threading
from collections import Counter, defaultdict

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*|[A-Za-z0-9_\-]+")
# CJK-aware: EverMemBench dialogues contain Chinese artifact names.
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text):
    text = text or ""
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    # character bigrams for CJK spans, which the latin tokenizer drops entirely
    if _CJK_RE.search(text):
        cjk = "".join(ch if _CJK_RE.match(ch) else " " for ch in text).split()
        for run in cjk:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
            if len(run) == 1:
                toks.append(run)
    return toks


def _best_gpu(min_free_mib=8000):
    """Return the GPU index with the most free memory (or 0 if unknown).

    `min_free_mib` is a floor, not a preference: on a shared cluster the
    freest card can still be nearly full, and loading onto it fails later with
    a CUDA OOM in the middle of a run rather than at startup. When nothing
    clears the floor we say so and fall back to CPU, which is slow but
    finishes.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
        best, best_free = 0, -1
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                free = int(parts[1])
                if free > best_free:
                    best, best_free = int(parts[0]), free
        if best_free < min_free_mib:
            print(f"[retrieval] no GPU has {min_free_mib} MiB free "
                  f"(best is cuda:{best} with {best_free} MiB); using CPU")
            return None
        return best
    except Exception:  # noqa: BLE001
        return 0


# Process-wide shared embedding model, loaded once under a lock. This both saves
# GPU memory (single model copy) and prevents a race where concurrent threads
# load SentenceTransformer simultaneously (which can leave weights on the meta
# device and crash with "Cannot copy out of meta tensor").
_shared_model = None
_shared_model_lock = threading.Lock()
_shared_reranker = None
_shared_reranker_lock = threading.Lock()


def _resolve_device(device=None):
    import torch
    if device:
        return device
    dev = config.EMBED_DEVICE
    if dev == "auto":
        if torch.cuda.is_available():
            g = _best_gpu()
            return "cpu" if g is None else f"cuda:{g}"
        return "cpu"
    if dev in ("cpu", "cuda") or dev.startswith("cuda:"):
        return dev
    return "cpu"


def get_embedding_model(model_path=None, device=None):
    """Return the shared SentenceTransformer, loading it once (thread-safe)."""
    global _shared_model
    model_path = model_path or config.BGE_MODEL_PATH
    if _shared_model is not None:
        return _shared_model
    with _shared_model_lock:
        if _shared_model is not None:
            return _shared_model
        import torch
        from sentence_transformers import SentenceTransformer
        dev = _resolve_device(device)
        kwargs = {}
        if dev.startswith("cuda"):
            kwargs["model_kwargs"] = {"dtype": torch.float16}
        _shared_model = SentenceTransformer(model_path, device=dev, **kwargs)
        _shared_model.eval()
        return _shared_model


def get_reranker():
    """Return a shared CrossEncoder reranker, or None if unavailable.

    A cross-encoder scores (query, passage) jointly and is far more accurate
    than bi-encoder cosine at the top of the list. This is the single biggest
    precision win available for a fixed context budget.
    """
    global _shared_reranker
    if _shared_reranker is not None:
        return _shared_reranker if _shared_reranker is not False else None
    if not config.USE_RERANK or not config.RERANKER_PATH:
        return None
    with _shared_reranker_lock:
        if _shared_reranker is not None:
            return _shared_reranker if _shared_reranker is not False else None
        try:
            import torch
            from sentence_transformers import CrossEncoder
            dev = _resolve_device()
            kwargs = {}
            if dev.startswith("cuda"):
                kwargs["automodel_args"] = {"torch_dtype": torch.float16}
            _shared_reranker = CrossEncoder(config.RERANKER_PATH, device=dev,
                                            max_length=512, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] reranker unavailable ({e}); using fusion order")
            _shared_reranker = False
            return None
        return _shared_reranker


def rerank(query, candidates, top_k, text_fn=None):
    """Cross-encoder rerank a candidate list of (score, message).

    Falls back to the incoming order when no reranker is configured.
    """
    if not candidates:
        return []
    model = get_reranker()
    if model is None:
        return candidates[:top_k]
    text_fn = text_fn or (lambda m: m.get("text", ""))
    pairs = [(query, text_fn(m)[:2000]) for _, m in candidates]
    try:
        scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        if "out of memory" in str(e).lower():
            # A co-tenant filled the card mid-run. Free what we can and retry
            # once with a small batch; a slow answer beats losing the question.
            try:
                import torch
                torch.cuda.empty_cache()
                scores = model.predict(pairs, batch_size=8,
                                       show_progress_bar=False)
            except Exception:  # noqa: BLE001
                print("[retrieval] reranker OOM; falling back to fusion order")
                return candidates[:top_k]
        else:
            return candidates[:top_k]
    ranked = sorted(zip(scores, [m for _, m in candidates]),
                    key=lambda x: -float(x[0]))
    return [(float(s), m) for s, m in ranked[:top_k]]


# ---------------------------------------------------------------------------
# BM25 (vectorised inverted index)
# ---------------------------------------------------------------------------
class BM25:
    """BM25 over an inverted index.

    The original implementation scanned every document for every query, which
    is O(N*|q|) per question over a 30k-message corpus. This version walks only
    the postings of query terms.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.messages = []
        self.doc_len = []
        self.avgdl = 0.0
        self.idf = {}
        self.postings = defaultdict(list)   # term -> [(doc_idx, tf), ...]
        self.N = 0

    def index(self, messages):
        self.messages = messages
        self.postings = defaultdict(list)
        self.doc_len = []
        df = Counter()
        for i, m in enumerate(messages):
            toks = tokenize(m.get("text"))
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            for t, c in tf.items():
                self.postings[t].append((i, c))
            df.update(tf.keys())
        self.N = len(messages)
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        q = tokenize(query)
        if not q:
            return []
        scores = defaultdict(float)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(s, self.messages[i]) for i, s in ranked]


# ---------------------------------------------------------------------------
# Dense (bge-m3) with disk cache
# ---------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, model_path=None, device=None,
                 batch_size=None, cache_dir=None):
        self.model_path = model_path or config.BGE_MODEL_PATH
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self.cache_dir = cache_dir or config.EMBED_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.messages = []
        self._matrix = None
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._model = get_embedding_model(self.model_path, self._device)
        return self._model

    def _cache_path(self, corpus_key):
        return self.cache_dir / f"{corpus_key}.npy"

    def index(self, messages, corpus_key=None):
        self.messages = messages
        key = corpus_key or hashlib_messages(messages)
        path = self._cache_path(key)
        if path.exists():
            self._matrix = np.load(path)
            if len(self._matrix) == len(messages):
                return
        model = self._load_model()
        # Embed messages WITH speaker/time context so the vector carries
        # attribution, not just topic. Plain body text made "who said X" and
        # date-anchored questions near-unretrievable.
        texts = [_embed_text(m) for m in messages]
        self._matrix = model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        np.save(path, self._matrix)

    def embed_query(self, query):
        model = self._load_model()
        v = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].astype(np.float32)

    def search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        if self._matrix is None:
            raise RuntimeError("DenseRetriever not indexed")
        qv = self.embed_query(query)
        sims = self._matrix @ qv
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        order = idx[np.argsort(-sims[idx])]
        return [(float(sims[i]), self.messages[i]) for i in order]


def _embed_text(m):
    """Text used for dense indexing: speaker + date + body."""
    ts = m.get("ts")
    date = ts.strftime("%Y-%m-%d") if ts else ""
    author = m.get("author") or ""
    role = m.get("role") or ""
    head = f"{date} {author}"
    if role:
        head += f" ({role})"
    return f"{head}: {m.get('text', '')}"


def hashlib_messages(messages):
    import hashlib
    h = hashlib.sha1()
    for m in messages[:200]:  # hash a prefix + length for a stable cheap key
        h.update((m["id"] or "").encode())
    h.update(str(len(messages)).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Hybrid (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists, k=None, weights=None):
    """Fuse multiple ranked lists of (score, message) via weighted RRF."""
    k = k if k is not None else config.RRF_K
    acc = {}
    weights = weights or [1.0] * len(ranked_lists)
    for w, ranked in zip(weights, ranked_lists):
        for rank, (_score, msg) in enumerate(ranked):
            acc[msg["id"]] = acc.get(msg["id"], 0.0) + w / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, dense=None, top_k=None):
        self.bm25 = BM25()
        self.dense = dense or DenseRetriever()
        self.top_k = top_k or config.TOP_K
        self.messages = []
        self.msg_by_id = {}

    def index(self, messages, corpus_key=None):
        self.messages = messages
        self.msg_by_id = {m["id"]: m for m in messages}
        self.bm25.index(messages)
        self.dense.index(messages, corpus_key=corpus_key)

    def search(self, query, top_k=None, depth=None, lexical_only=False):
        """Fuse BM25 and dense rankings over a wide candidate pool.

        `lexical_only` collapses this to a single BM25 ranking, which is what a
        plain lexical baseline does. It exists for the `noseed` ablation: the
        wide two-retriever pool is otherwise outside every ablation block, and
        so is a candidate for the unattributed residual gain.
        """
        top_k = top_k or self.top_k
        depth = depth or max(top_k * 3, config.CANDIDATE_K)
        r_bm25 = self.bm25.search(query, top_k=depth)
        if lexical_only:
            fused = [(mid, s) for mid, s in r_bm25]
        else:
            r_dense = self.dense.search(query, top_k=depth)
            fused = rrf_fuse([r_bm25, r_dense])
        out = []
        for mid, score in fused[:top_k]:
            m = self.msg_by_id.get(mid)
            if m is not None:
                out.append((score, m))
        return out
