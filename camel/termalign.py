"""Contextual Term Grounding (CTG): typed, context-aware terminology alignment.

Motivation
----------
Speakers adapt vocabulary to their audience, so one concept surfaces under
different role-indexed forms ("finance workflows" from a CFO vs "Finance Ops"
in the evidence). The original mechanism scored such pairs by bi-encoder cosine
against a threshold, which is the wrong instrument twice over:

  * it is *brittle*: at tau=0.86 the mechanism was completely inert -- the
    -noterm ablation was byte-identical to the full system on all 90
    GroupMemBench questions, i.e. it contributed exactly nothing;
  * it is *untyped*: cosine cannot separate "same concept" from "related
    concept". "Finance" and "Finance Ops" score high but stand in a
    broader/narrower relation, so treating them as aliases injects wrong
    evidence.

Cosine optimises overall semantic proximity, not co-reference. We therefore
demote similarity to a *candidate generator* and pose alignment as contextual
relation inference.

Formulation
-----------
Stage 1 -- high-recall candidate retrieval (no hard threshold):

    C(x) = TopK_{y in V} cos(e_x, e_y)

Stage 2 -- typed relation inference, conditioned on the conversational context
c in which the term was used:

    r(x, y, c) in {equivalent, broader, narrower, related, unrelated}

realised as *bidirectional entailment*. Rather than one scalar similarity we
ask two directional questions and combine them:

    Alias(x, y) = 1[ P(x=>y | c) > tau_e  AND  P(y=>x | c) > tau_e ]

    x=>y, y=>x   ->  equivalent   (mutual entailment: aliases)
    x=>y, not    ->  narrower     (x is a special case of y)
    not,   y=>x  ->  broader
    neither      ->  related / unrelated (by score)

Only `equivalent` licenses alias expansion; `broader`/`narrower` are recorded
but not expanded, which is what prevents the "Finance" / "Finance Ops" error.

Backends
--------
`nli`      a natural-language-inference cross-encoder; the entailment
           probability is read directly per direction. Preferred.
`reranker` a relevance cross-encoder (e.g. bge-reranker-v2-m3), scoring a
           directional "does A refer to B" query. A strong zero-shot baseline
           when no NLI model is available.
`cosine`   the original thresholded bi-encoder, kept as the tuned baseline
           (SimAlign_tuned) for the ablation table.

Everything here is zero-shot: no training data, no fine-tuning.
"""
import math
import re
import threading

import numpy as np

from . import config

# Relation labels
EQUIVALENT = "equivalent"
BROADER = "broader"
NARROWER = "narrower"
RELATED = "related"
UNRELATED = "unrelated"

_model_lock = threading.Lock()
_nli = None
_nli_failed = False


def _get_nli():
    """Load the NLI cross-encoder once, or return None if unavailable."""
    global _nli, _nli_failed
    if _nli is not None or _nli_failed:
        return _nli
    with _model_lock:
        if _nli is not None or _nli_failed:
            return _nli
        path = config.TERM_NLI_PATH
        if not path:
            _nli_failed = True
            return None
        try:
            from sentence_transformers import CrossEncoder
            from .retrieval import _resolve_device
            _nli = CrossEncoder(path, device=_resolve_device(), max_length=256)
        except Exception as e:  # noqa: BLE001
            print(f"[termalign] NLI model unavailable ({e}); "
                  f"falling back to backend='{config.TERM_BACKEND_FALLBACK}'")
            _nli_failed = True
            return None
        return _nli


def _entail_label_index(model):
    """Locate the 'entailment' column of an NLI head, tolerating label order."""
    cfg = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(cfg, "id2label", None) or {}
    for i, lab in id2label.items():
        if str(lab).lower().startswith("entail"):
            return int(i)
    # sentence-transformers NLI cross-encoders conventionally use
    # ['contradiction', 'entailment', 'neutral']
    return 1


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / max(e.sum(), 1e-12)


def _premise(term, ctx, role=None):
    """Render one side of the pair with its conversational context."""
    bits = []
    if role:
        bits.append(f"A {role} says:")
    if ctx:
        bits.append(f'"{ctx.strip()[:220]}"')
    bits.append(f"This concerns {term}.")
    return " ".join(bits)


def _hypothesis(term):
    return f"This concerns {term}."


class TermAligner:
    """Two-stage contextual terminology alignment.

    Usage:
        al = TermAligner(entities, embeddings)
        al.build()                      # candidate retrieval + relation typing
        al.aliases_of("finance workflows")   -> {"finance ops", ...}
    """

    def __init__(self, entities, embeddings, contexts=None, roles=None,
                 backend=None, top_k=None, tau=None):
        self.entities = list(entities)
        self.emb = embeddings
        self.contexts = contexts or {}    # entity -> a representative utterance
        self.roles = roles or {}          # entity -> a speaker role that used it
        self.backend = (backend or config.TERM_BACKEND).lower()
        self.top_k = top_k if top_k is not None else config.TERM_TOPK
        self.tau = tau if tau is not None else config.TERM_TAU
        # Ablation switch: when True, a single entailment direction suffices,
        # which is what conflates broader/narrower with equivalent.
        self.one_way = False
        self.relations = {}               # (a, b) -> label
        self.aliases = {}                 # entity -> set of equivalent entities
        self.stats = {"candidates": 0, EQUIVALENT: 0, BROADER: 0,
                      NARROWER: 0, RELATED: 0, UNRELATED: 0}

    # -- stage 1: high-recall candidates --------------------------------
    def candidates(self):
        """Top-k nearest neighbours per entity. No hard threshold: recall only."""
        n = len(self.entities)
        if n < 2:
            return []
        out, seen = [], set()
        block = 512
        floor = config.TERM_CAND_FLOOR
        for start in range(0, n, block):
            sims = self.emb[start:start + block] @ self.emb.T
            for bi in range(sims.shape[0]):
                i = start + bi
                row = sims[bi]
                order = np.argsort(-row)[1:self.top_k + 1]
                for j in order:
                    j = int(j)
                    if j == i or row[j] < floor:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((self.entities[i], self.entities[j], float(row[j])))
        self.stats["candidates"] = len(out)
        return out

    # -- stage 2: typed relation inference ------------------------------
    def _entail_scores(self, pairs):
        """P(a => b) for each (a, b), using the configured backend."""
        if self.backend == "nli":
            model = _get_nli()
            if model is not None:
                idx = _entail_label_index(model)
                inputs = [(_premise(a, self.contexts.get(a), self.roles.get(a)),
                           _hypothesis(b)) for a, b in pairs]
                raw = model.predict(inputs, batch_size=64,
                                    show_progress_bar=False)
                raw = np.asarray(raw)
                if raw.ndim == 2:
                    return np.array([_softmax(r)[idx] for r in raw])
                return 1.0 / (1.0 + np.exp(-raw))    # single-logit head
            # fall through to the configured fallback
            self.backend = config.TERM_BACKEND_FALLBACK

        if self.backend == "reranker":
            from .retrieval import get_reranker
            model = get_reranker()
            if model is not None:
                # A relevance cross-encoder answers "is this document relevant
                # to this query", NOT "does A entail B". Naming the target term
                # inside the query text makes the target document trivially
                # relevant, so every pair scored high in both directions and
                # 7395/7395 candidates came back `equivalent`.
                #
                # Instead the query is the SOURCE term in its own context, and
                # the document is the TARGET term with its context. The score is
                # then a genuine directional "how well does b cover a", which
                # differs between the two directions for hypernym pairs.
                inputs = []
                for a, b in pairs:
                    ca = (self.contexts.get(a) or "").strip()[:160]
                    cb = (self.contexts.get(b) or "").strip()[:160]
                    q = f"{a}: {ca}" if ca else a
                    d = f"{b}: {cb}" if cb else b
                    inputs.append((q, d))
                raw = np.asarray(model.predict(inputs, batch_size=64,
                                               show_progress_bar=False))
                if raw.ndim == 2:
                    raw = raw[:, -1]
                return 1.0 / (1.0 + np.exp(-raw))
        return None    # caller falls back to cosine

    def build(self):
        cands = self.candidates()
        if not cands:
            return self
        if self.backend == "cosine":
            # SimAlign_tuned: the strongest thresholded-similarity baseline.
            for a, b, s in cands:
                lab = EQUIVALENT if s >= config.ALIAS_SIM else RELATED
                self._record(a, b, lab)
            return self

        fwd = self._entail_scores([(a, b) for a, b, _ in cands])
        bwd = self._entail_scores([(b, a) for a, b, _ in cands])
        if fwd is None or bwd is None:
            # no cross-encoder available: degrade to the cosine baseline
            for a, b, s in cands:
                self._record(a, b, EQUIVALENT if s >= config.ALIAS_SIM else RELATED)
            return self

        # Calibrate tau to the observed score distribution. A fixed absolute
        # threshold assumes the backend emits calibrated probabilities, which a
        # relevance cross-encoder does not: at tau=0.5 every candidate scored
        # above it and 100% of pairs came back `equivalent`. Using a quantile of
        # the actual scores keeps the decision meaningful whatever the backend's
        # scale, and `equivalent` stays a minority label by construction.
        tau = self.tau
        if config.TERM_CALIBRATE:
            allsc = np.concatenate([np.asarray(fwd), np.asarray(bwd)])
            q = float(np.quantile(allsc, config.TERM_QUANTILE))
            if np.ptp(allsc) < 1e-6:
                print("[termalign] WARNING: backend returned a constant score; "
                      "relation inference is uninformative. Falling back to "
                      "the cosine baseline.")
                for a, b, s in cands:
                    self._record(a, b,
                                 EQUIVALENT if s >= config.ALIAS_SIM else RELATED)
                return self
            tau = q
            self.tau = tau

        for (a, b, s), pf, pb in zip(cands, fwd, bwd):
            self._record(a, b, self._label(float(pf), float(pb)))

        # Degeneracy guard: if essentially everything (or nothing) is
        # `equivalent`, the relation signal is not real. Say so loudly rather
        # than shipping an alias table that links every term to every other.
        n = max(1, self.stats["candidates"])
        frac = self.stats.get(EQUIVALENT, 0) / n
        if frac > 0.60 or frac < 0.002:
            print(f"[termalign] WARNING: {frac:.1%} of candidates labelled "
                  f"'{EQUIVALENT}' -- the relation model looks degenerate. "
                  f"Check term_align.backend / tau.")
        return self

    def _label(self, p_fwd, p_bwd):
        """Bidirectional entailment -> typed relation."""
        f, r = p_fwd >= self.tau, p_bwd >= self.tau
        if self.one_way:
            # Ablation: accept on either direction. This is precisely what
            # makes "Finance" and "Finance Ops" look equivalent.
            return EQUIVALENT if (f or r) else (
                RELATED if max(p_fwd, p_bwd) >= self.tau * config.TERM_RELATED_FRAC
                else UNRELATED)
        if f and r:
            return EQUIVALENT
        if f:
            return NARROWER      # a entails b: a is the special case
        if r:
            return BROADER
        if max(p_fwd, p_bwd) >= self.tau * config.TERM_RELATED_FRAC:
            return RELATED
        return UNRELATED

    def _record(self, a, b, label):
        self.relations[(a, b)] = label
        self.stats[label] = self.stats.get(label, 0) + 1
        # ONLY mutual entailment licenses alias expansion. broader/narrower are
        # kept for analysis but must not expand, or "Finance" would pull in all
        # of "Finance Ops" and inject wrong evidence.
        if label == EQUIVALENT:
            self.aliases.setdefault(a, set()).add(b)
            self.aliases.setdefault(b, set()).add(a)

    def aliases_of(self, term):
        return self.aliases.get(term, set())

    def summary(self):
        s = self.stats
        return (f"backend={self.backend} candidates={s['candidates']} "
                f"equivalent={s.get(EQUIVALENT, 0)} broader={s.get(BROADER, 0)} "
                f"narrower={s.get(NARROWER, 0)} related={s.get(RELATED, 0)} "
                f"unrelated={s.get(UNRELATED, 0)}")
