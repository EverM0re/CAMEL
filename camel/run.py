"""Experiment driver: run GroupMemBench and EverMemBench, cache, aggregate.

Usage:
    python -m camel.run groupmem --domains Finance --methods bm25 CAMEL \
        --limit 40 --run-id pilot1
    python -m camel.run evermem --topics 01 --methods bm25 CAMEL --limit 60
    python -m camel.run sweep --bench evermem --param top_k --values 10 20 30
    python -m camel.run aggregate --run-id pilot1
    python -m camel.run smoke

Results are written to results/<run_id>/<bench>/<method>__<key>.jsonl and are
resumable: any question already answered for a (method, qid) is skipped on
re-run.  Question processing is parallelised with a thread pool (the LLM API is
the bottleneck).
"""
import argparse
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, data
from .engine import Corpus, METHODS
from .qa import (Answerer, Judge, parse_groupmem_verdict, parse_evermem_verdict,
                 extract_final)

_write_lock = threading.Lock()
_ACTIVE_TOP_K = None    # --top-k for this invocation, recorded in the manifest


def _run_root(run_id=None):
    return config.RESULTS_DIR / run_id if run_id else config.RESULTS_DIR


def _result_path(bench, method, key, run_id=None):
    d = _run_root(run_id) / bench
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{method}__{key}.jsonl"


def _load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["qid"]] = r
    return done


def _append_records(path, records):
    with _write_lock:
        with path.open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample(questions, limit, seed=0, stratify_key=None):
    """Take a reproducible subsample, stratified by category when asked.

    A pilot must not silently over-represent whichever category happens to come
    first in the file, otherwise the pilot's averages are not comparable with
    the full run's.
    """
    if not limit or limit >= len(questions):
        return questions
    rnd = random.Random(seed)
    if not stratify_key:
        out = list(questions)
        rnd.shuffle(out)
        return out[:limit]
    buckets = defaultdict(list)
    for q in questions:
        buckets[q.get(stratify_key, "?")].append(q)
    for v in buckets.values():
        rnd.shuffle(v)
    out, keys = [], sorted(buckets)
    i = 0
    while len(out) < limit and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out[:limit]


def _write_manifest(run_id, payload):
    root = _run_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("runs", []).append(payload)
    existing["run_id"] = run_id
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


# Bump when a change should invalidate cached answers or must be visible in
# the manifest. r10 silently re-ran old code because nothing in the output
# identified the code version.
CODE_VERSION = "2026-09-08b-hint-telemetry"


def _config_snapshot():
    return {
        "code_version": CODE_VERSION,
        "top_k": config.TOP_K, "candidate_k": config.CANDIDATE_K,
        "top_k_override": _ACTIVE_TOP_K,
        "graph_hops": config.GRAPH_HOPS, "graph_anchors": config.GRAPH_ANCHORS,
        "graph_budget_frac": config.GRAPH_BUDGET_FRAC,
        "max_context_chars": config.MAX_CONTEXT_CHARS,
        "rrf_k": config.RRF_K, "alias_sim": config.ALIAS_SIM,
        "decompose": config.DECOMPOSE, "max_subqueries": config.MAX_SUBQUERIES,
        "use_rerank": config.USE_RERANK, "reranker": config.RERANKER_PATH,
        "rerank_depth": config.RERANK_DEPTH,
        "use_timeline": config.USE_TIMELINE,
        "answer_model": config.ANSWER_MODEL, "judge_model": config.JUDGE_MODEL,
    }


# ---------------------------------------------------------------------------
def _drop_skipped(methods):
    """Remove methods named in CAMEL_SKIP_METHODS.

    RAPTOR's Gaussian-mixture tree construction exhausts GPU memory on large
    corpora and on a shared device, and it does so AFTER spending several
    minutes building; skipping it up front avoids paying that cost once per
    benchmark for a result we already report as unavailable.
    """
    skip = {m.strip() for m in
            os.environ.get("CAMEL_SKIP_METHODS", "").split(",") if m.strip()}
    if not skip:
        return methods
    kept = [m for m in methods if m not in skip]
    for m in methods:
        if m in skip:
            print(f"[skip] {m}: listed in CAMEL_SKIP_METHODS", flush=True)
    return kept


def run_groupmem(domains=None, methods=None, qtypes=None, top_k=None, workers=16,
                 limit=None, run_id=None, seed=0):
    from .llm import LLMClient
    global _ACTIVE_TOP_K
    _ACTIVE_TOP_K = top_k
    domains = domains or config.GROUP_DOMAINS
    methods = methods or METHODS
    qtypes = qtypes or config.GROUP_QTYPES
    # GroupMemBench has no gold-evidence annotation, so an "oracle" method is
    # meaningless here (it would return an empty context and answer by guess).
    methods = [m for m in methods if m != "oracle"]
    methods = _drop_skipped(methods)

    for domain in domains:
        print(f"\n=== GroupMemBench / {domain} ===", flush=True)
        t0 = time.time()
        messages, _ = data.load_groupmem_domain(domain)
        corpus = Corpus(messages, corpus_key=f"groupmem_{domain}")
        llm = LLMClient(model=config.ANSWER_MODEL, max_tokens=512)
        build_flat = "flat_mem" in methods
        corpus.build(use_graph=True, use_alias=True, build_flat=build_flat, llm=llm)
        questions = [q for q in data.load_groupmem_questions(domain, qtype=None)
                     if q["qtype"] in qtypes]
        questions = _sample(questions, limit, seed=seed, stratify_key="qtype")
        print(f"  {len(questions)} questions, {len(messages)} messages "
              f"(index {time.time() - t0:.0f}s)", flush=True)

        for method in methods:
            path = _result_path("groupmem", method, domain, run_id)
            done = _load_done(path)
            todo = [q for q in questions if q["id"] not in done]
            if not todo:
                print(f"  [{method}] {len(done)}/{len(questions)} done", flush=True)
                continue
            t1 = time.time()

            def work(q, _m=method):
                a, j = _thread_qa()
                asker = q.get("asking_user_id")
                ctx = corpus.answer_context(_m, q["question"], asker=asker,
                                            q_meta={"qtype": q["qtype"]},
                                            top_k=top_k)
                ans = a.answer_groupmem(q["question"], ctx, asker=asker)
                final = extract_final(ans)
                verdict = j.judge_groupmem(q["question"], q["answer"], final)
                v = parse_groupmem_verdict(verdict)
                rec = {
                    "qid": q["id"], "qtype": q["qtype"], "method": _m,
                    "question": q["question"],
                    "asker": asker, "answer": final, "answer_raw": ans,
                    "gold": q["answer"],
                    "verdict": v, "unclear": v is None,
                    "verdict_raw": verdict,
                }
                rec.update(_cost_metrics(corpus, _m, q["question"],
                                         {"qtype": q["qtype"]}, top_k,
                                         asker=asker, ctx=ctx))
                # Which context augmentations actually fired. Without this a
                # null result is indistinguishable from a mechanism that never
                # triggered -- exactly the ambiguity that cost us r11.
                rec["hint_date_arith"] = "DATE ARITHMETIC" in (ctx or "")
                rec["hint_date_resolved"] = "RESOLVED DATES" in (ctx or "")
                rec["has_profile"] = "USER PROFILE" in (ctx or "")
                return rec

            _drain(path, todo, work, workers)
            n = len(_load_done(path))
            print(f"  [{method}] {n}/{len(questions)} done "
                  f"({time.time() - t1:.0f}s)", flush=True)

    _write_manifest(run_id, {"bench": "groupmem", "domains": domains,
                             "methods": methods, "qtypes": qtypes,
                             "limit": limit, "seed": seed,
                             "config": _config_snapshot()})


# ---------------------------------------------------------------------------
def run_evermem(topics=None, methods=None, top_k=None, workers=16, limit=None,
                run_id=None, seed=0):
    from .llm import LLMClient
    global _ACTIVE_TOP_K
    _ACTIVE_TOP_K = top_k
    topics = topics or config.EVER_TOPICS
    methods = _drop_skipped(methods or METHODS)

    for topic in topics:
        print(f"\n=== EverMemBench / topic {topic} ===", flush=True)
        t0 = time.time()
        messages, profiles = data.load_evermem_topic(topic)
        corpus = Corpus(messages, profiles_by_name=profiles,
                        corpus_key=f"evermem_{topic}")
        llm = LLMClient(model=config.ANSWER_MODEL, max_tokens=512)
        build_flat = "flat_mem" in methods
        corpus.build(use_graph=True, use_alias=True, build_flat=build_flat, llm=llm)
        questions = data.load_evermem_questions(topic)
        questions = _sample(questions, limit, seed=seed, stratify_key="category")
        print(f"  {len(questions)} questions, {len(messages)} messages "
              f"(index {time.time() - t0:.0f}s)", flush=True)

        for method in methods:
            path = _result_path("evermem", method, topic, run_id)
            done = _load_done(path)
            todo = [q for q in questions if q["id"] not in done]
            if not todo:
                print(f"  [{method}] {len(done)}/{len(questions)} done", flush=True)
                continue
            t1 = time.time()

            def work(q, _m=method):
                a, j = _thread_qa()
                q_meta = {"reference": q.get("reference"),
                          "topic_id": topic, "category": q["category"],
                          "dimension": q["dimension"]}
                ctx = corpus.answer_context(_m, q["question"], asker=None,
                                            q_meta=q_meta, top_k=top_k)
                ans = a.answer_evermem(q["question"], ctx, q.get("options"))
                if q.get("options"):
                    # Multiple choice is graded by exact letter match, as the
                    # benchmark specifies. Sending it to the LLM judge would
                    # double the API cost and add grader noise to a comparison
                    # that is already deterministic.
                    from .qa import normalize_mc
                    ans = normalize_mc(ans)
                    gold = normalize_mc(q["answer"]) or str(q["answer"]).strip()
                    v = 1 if (ans and ans == gold) else 0
                    verdict = f"exact-match: pred={ans!r} gold={gold!r}"
                else:
                    verdict = j.judge_evermem(q["question"], q["answer"], ans)
                    v = parse_evermem_verdict(verdict)
                rec = {
                    "qid": q["id"], "category": q["category"],
                    "dimension": q["dimension"], "method": _m,
                    "answer": ans, "gold": q["answer"],
                    "mc": bool(q.get("options")),
                    "verdict": v, "unclear": v is None,
                    "verdict_raw": verdict,
                }
                rec.update(_retrieval_metrics(corpus, _m, q["question"],
                                              q_meta, top_k))
                rec.update(_cost_metrics(corpus, _m, q["question"], q_meta,
                                         top_k, ctx=ctx))
                rec["question"] = q["question"]
                rec["hint_date_arith"] = "DATE ARITHMETIC" in (ctx or "")
                rec["hint_date_resolved"] = "RESOLVED DATES" in (ctx or "")
                rec["has_profile"] = "USER PROFILE" in (ctx or "")
                return rec

            _drain(path, todo, work, workers)
            n = len(_load_done(path))
            print(f"  [{method}] {n}/{len(questions)} done "
                  f"({time.time() - t1:.0f}s)", flush=True)

    _write_manifest(run_id, {"bench": "evermem", "topics": topics,
                             "methods": methods, "limit": limit, "seed": seed,
                             "config": _config_snapshot()})


# ---------------------------------------------------------------------------
def run_socialmem(networks=None, methods=None, top_k=None, workers=16,
                  limit=None, run_id=None, seed=0):
    """SocialMemBench: one corpus per social network.

    Memory persists within a network and resets between networks, which is the
    benchmark's own protocol, so a network is the natural corpus unit.
    """
    from .llm import LLMClient
    global _ACTIVE_TOP_K
    _ACTIVE_TOP_K = top_k
    networks = networks or data.list_socialmem_networks()
    methods = _drop_skipped(methods or METHODS)

    for net in networks:
        print(f"\n=== SocialMemBench / {net} ===", flush=True)
        t0 = time.time()
        messages, profiles = data.load_socialmem_network(net)
        if not messages:
            print(f"  (no messages for {net}; skipping)", flush=True)
            continue
        corpus = Corpus(messages, profiles_by_name=profiles,
                        corpus_key=f"social_{net}")
        llm = LLMClient(model=config.ANSWER_MODEL, max_tokens=512)
        corpus.build(use_graph=True, use_alias=True,
                     build_flat="flat_mem" in methods, llm=llm)
        questions = _sample(data.load_socialmem_questions(net), limit,
                            seed=seed, stratify_key="qtype")
        print(f"  {len(questions)} questions, {len(messages)} messages "
              f"(index {time.time() - t0:.0f}s)", flush=True)

        for method in methods:
            path = _result_path("socialmem", method, net, run_id)
            done = _load_done(path)
            todo = [q for q in questions if q["id"] not in done]
            if not todo:
                print(f"  [{method}] {len(done)}/{len(questions)} done", flush=True)
                continue
            t1 = time.time()

            def work(q, _m=method):
                a, j = _thread_qa()
                q_meta = {"reference": q.get("reference"), "qtype": q["qtype"],
                          "category": q["category"], "dimension": q["dimension"]}
                ctx = corpus.answer_context(_m, q["question"], asker=None,
                                            q_meta=q_meta, top_k=top_k)
                ans = a.answer_evermem(q["question"], ctx, q.get("options"))
                if q.get("options"):
                    from .qa import normalize_mc
                    ans = normalize_mc(ans)
                    gold = normalize_mc(q["answer"]) or str(q["answer"]).strip()
                    v = 1 if (ans and ans == gold) else 0
                    verdict = f"exact-match: pred={ans!r} gold={gold!r}"
                else:
                    verdict = j.judge_evermem(q["question"], q["answer"], ans)
                    v = parse_evermem_verdict(verdict)
                rec = {"qid": q["id"], "category": q["category"],
                       "qtype": q["qtype"], "dimension": q["dimension"],
                       "method": _m, "answer": ans, "gold": q["answer"],
                       "mc": bool(q.get("options")), "verdict": v,
                       "unclear": v is None, "verdict_raw": verdict}
                rec.update(_retrieval_metrics(corpus, _m, q["question"],
                                              q_meta, top_k))
                rec.update(_cost_metrics(corpus, _m, q["question"], q_meta,
                                         top_k, ctx=ctx))
                rec["question"] = q["question"]
                rec["hint_date_arith"] = "DATE ARITHMETIC" in (ctx or "")
                rec["hint_date_resolved"] = "RESOLVED DATES" in (ctx or "")
                rec["has_profile"] = "USER PROFILE" in (ctx or "")
                return rec

            _drain(path, todo, work, workers)
            print(f"  [{method}] {len(_load_done(path))}/{len(questions)} done "
                  f"({time.time() - t1:.0f}s)", flush=True)

    _write_manifest(run_id, {"bench": "socialmem", "networks": networks,
                             "methods": methods, "limit": limit, "seed": seed,
                             "config": _config_snapshot()})


def _cost_metrics(corpus, method, question, q_meta, top_k, asker=None, ctx=None):
    """Retrieval latency and context size for one question.

    Efficiency is a first-class axis for memory systems -- both benchmark
    papers report ingestion cost -- but wall-clock per method conflates
    retrieval with the LLM call. This times the retrieval step alone and
    records the context actually handed to the answer model.
    """
    out = {}
    if ctx is not None:
        out["ctx_chars"] = len(ctx)
        # ~4 chars/token is the usual English approximation; exact counts would
        # need the provider's tokenizer, which differs per backbone.
        out["ctx_tokens_est"] = round(len(ctx) / 4)
    t0 = time.perf_counter()
    try:
        ev = corpus.retrieve(method, question, asker=asker, q_meta=q_meta,
                             top_k=top_k)
        out["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out["n_evidence"] = len(ev)
    except Exception:  # noqa: BLE001
        pass
    return out


def _retrieval_metrics(corpus, method, question, q_meta, top_k, asker=None):
    """Recall/precision of the retrieved window against annotated gold evidence.

    Accuracy alone cannot separate "the evidence never arrived" from "the
    evidence arrived and the model misread it". Reporting recall@k next to
    accuracy makes that split explicit, and it is the metric the multi-hop
    gap (CAMEL 13.9 vs oracle 99.1) actually turns on.
    """
    gold = corpus.gold_ids(q_meta)
    if not gold:
        return {}
    try:
        ev = corpus.retrieve(method, question, asker=asker, q_meta=q_meta,
                             top_k=top_k)
    except Exception:  # noqa: BLE001
        return {}
    got = {m["id"] for _, m in ev}
    hit = len(got & gold)
    return {
        "gold_n": len(gold),
        "retrieved_n": len(got),
        "recall": round(hit / len(gold), 4),
        "precision": round(hit / max(1, len(got)), 4),
        "full_recall": int(hit == len(gold)),
    }


def _drain(path, todo, work, workers):
    """Run `work` over `todo` in a pool, flushing results incrementally.

    Cancels outstanding work on Ctrl+C or a fatal API condition. Without the
    explicit cancel, ThreadPoolExecutor's exit waits for every queued future,
    so an interrupt appeared to hang while hundreds of doomed calls retried.
    """
    from .llm import FatalLLMError, fatal_reason
    ex = ThreadPoolExecutor(max_workers=workers)
    futs, batch, n_err, aborted = [], [], 0, None
    try:
        futs = [ex.submit(work, q) for q in todo]
        for fu in as_completed(futs):
            try:
                batch.append(fu.result())
            except FatalLLMError as e:
                aborted = str(e)
                break
            except Exception as e:  # noqa: BLE001
                n_err += 1
                if n_err <= 3:
                    print(f"    ! question failed: {e}", flush=True)
            if len(batch) >= 100:
                _append_records(path, batch)
                batch = []
    except KeyboardInterrupt:
        aborted = "interrupted by user"
        print("\n    ! interrupt received -- cancelling queued work", flush=True)
    finally:
        if aborted:
            for f in futs:
                f.cancel()
        if batch:
            _append_records(path, batch)   # never lose completed answers
        ex.shutdown(wait=not aborted)
    if n_err:
        print(f"    ! {n_err} question(s) failed", flush=True)
    if aborted or fatal_reason():
        raise _RunAborted(aborted or fatal_reason())


class _RunAborted(RuntimeError):
    """Signals that the whole run should stop, with partial results saved."""


# thread-local answerer/judge (each with its own LLM client)
_thread_local = threading.local()


def _thread_qa():
    if not hasattr(_thread_local, "qa"):
        from .llm import LLMClient
        _thread_local.qa = (
            Answerer(llm=LLMClient(model=config.ANSWER_MODEL, max_tokens=512)),
            Judge(llm=LLMClient(model=config.JUDGE_MODEL, max_tokens=128,
                              api=config.JUDGE_API,
                              base_url=config.JUDGE_BASE_URL,
                              api_key=config.JUDGE_API_KEY)),
        )
    return _thread_local.qa


# ---------------------------------------------------------------------------
def aggregate(run_id=None, quiet=False):
    root = _run_root(run_id)
    out = {"groupmem": {}, "evermem": {}, "socialmem": {}}
    per_cat = {}
    for bench, cat_key in (("groupmem", "qtype"), ("evermem", "category"),
                           ("socialmem", "category")):
        base = root / bench
        if not base.exists():
            continue
        stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # n, corr, unclear
        retr = defaultdict(lambda: [0, 0.0, 0.0, 0])  # n, recall, prec, full
        for f in sorted(base.glob("*.jsonl")):
            method = f.name.split("__")[0]
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cat = r.get(cat_key, "?")
                v = r.get("verdict")
                s = stats[method][cat]
                if v is None:
                    s[2] += 1          # unclear -> excluded from denominator
                    continue
                s[0] += 1
                s[1] += v
                if r.get("recall") is not None:
                    a = retr[method]
                    a[0] += 1; a[1] += r["recall"]
                    a[2] += r.get("precision", 0.0); a[3] += r.get("full_recall", 0)
        if not stats:
            continue
        cats = sorted({c for m in stats for c in stats[m]})
        if not quiet:
            print(f"\n===== {bench.upper()} accuracy by method =====")
            print(f"{'method':24s}" + "".join(f"{c[:11]:>12s}" for c in cats)
                  + f"{'AVG':>8s}{'n':>7s}{'recall':>9s}{'fullrec':>9s}")
        rows = {}
        for method in sorted(stats):
            row, tot_n, tot_c = "", 0, 0
            cat_acc = {}
            for c in cats:
                n, corr, _u = stats[method].get(c, [0, 0, 0])
                tot_n += n
                tot_c += corr
                acc = corr / n * 100 if n else 0.0
                cat_acc[c] = round(acc, 2)
                row += f"{acc:>12.1f}" if n else f"{'-':>12s}"
            avg = tot_c / tot_n * 100 if tot_n else 0.0
            if not quiet:
                rn, rr, _rp, rf = retr.get(method, [0, 0.0, 0.0, 0])
            extra = (f"{rr / rn * 100:>9.1f}{rf / rn * 100:>9.1f}") if rn else ""
            print(f"{method:24s}{row}{avg:>8.1f}{tot_n:>7d}{extra}")
            rows[method] = {"n": tot_n, "correct": tot_c, "acc": round(avg, 2),
                            "by_category": cat_acc}
            rn, rr, rp, rf = retr.get(method, [0, 0.0, 0.0, 0])
            if rn:
                rows[method].update({
                    "recall": round(rr / rn * 100, 2),
                    "precision": round(rp / rn * 100, 2),
                    "full_recall": round(rf / rn * 100, 2)})
        out[bench] = rows
        per_cat[bench] = {"categories": cats}

    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if not quiet:
        print(f"\nWrote {path}")
    return out


# ---------------------------------------------------------------------------
def sweep(bench, param, values, base_methods=None, run_id=None, **kw):
    """Hyper-parameter sweep: re-run one method under different config values."""
    env_map = {
        "top_k": "CAMEL_TOP_K",
        "candidate_k": "CAMEL_CANDIDATE_K",
        "graph_hops": "CAMEL_GRAPH_HOPS",
        "graph_anchors": "CAMEL_GRAPH_ANCHORS",
        "graph_budget_frac": "CAMEL_GRAPH_BUDGET_FRAC",
        "alias_sim": "CAMEL_ALIAS_SIM",
        "rrf_k": "CAMEL_RRF_K",
        "max_subqueries": "CAMEL_MAX_SUBQUERIES",
        "rerank_depth": "CAMEL_RERANK_DEPTH",
    }
    if param not in env_map:
        raise SystemExit(f"unknown sweep param {param}; choose from {sorted(env_map)}")
    base_methods = base_methods or ["camel"]
    # Each setting runs in a FRESH SUBPROCESS. Reloading the config module
    # in-process leaves already-imported modules holding stale references (and
    # resets process-wide singletons such as the embedding model), so the value
    # under test would not reliably take effect.
    import subprocess
    import sys
    for v in values:
        tag = f"{run_id or 'sweep'}__{param}_{v}"
        print(f"\n########## sweep {param}={v} -> {tag} ##########", flush=True)
        env = dict(os.environ)
        env[env_map[param]] = str(v)
        cmd = [sys.executable, "-m", "camel.run", bench,
               "--methods", *base_methods, "--run-id", tag,
               "--workers", str(kw.get("workers", 16))]
        if kw.get("limit"):
            cmd += ["--limit", str(kw["limit"])]
        if bench == "groupmem" and kw.get("domains"):
            cmd += ["--domains", *kw["domains"]]
        if bench == "evermem" and kw.get("topics"):
            cmd += ["--topics", *kw["topics"]]
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f"  ! sweep point {param}={v} exited with {rc}", flush=True)

    # roll the sweep points up into one comparison table
    rows = {}
    for v in values:
        tag = f"{run_id or 'sweep'}__{param}_{v}"
        p = _run_root(tag) / "summary.json"
        if p.exists():
            try:
                rows[str(v)] = json.loads(p.read_text()).get(bench, {})
            except json.JSONDecodeError:
                pass
    if rows:
        out = _run_root(f"{run_id or 'sweep'}__{param}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(
            json.dumps({"param": param, "bench": bench, "points": rows},
                       indent=2, ensure_ascii=False))
        print(f"\n===== sweep {param} =====")
        methods = sorted({m for r in rows.values() for m in r})
        print(f"{'value':>10s}" + "".join(f"{m[:16]:>18s}" for m in methods))
        for v, r in rows.items():
            print(f"{v:>10s}" + "".join(
                f"{r.get(m, {}).get('acc', float('nan')):>18.2f}" for m in methods))
        print(f"\nWrote {out / 'sweep.json'}")


# ---------------------------------------------------------------------------
def smoke():
    from .llm import LLMClient
    llm = LLMClient(model=config.ANSWER_MODEL, max_tokens=512)
    answerer = Answerer(llm=llm)
    judge = Judge(llm=LLMClient(model=config.JUDGE_MODEL, max_tokens=128,
                              api=config.JUDGE_API,
                              base_url=config.JUDGE_BASE_URL,
                              api_key=config.JUDGE_API_KEY))

    topic = "01"
    messages, profiles = data.load_evermem_topic(topic)
    corpus = Corpus(messages, profiles_by_name=profiles, corpus_key=f"evermem_{topic}")
    corpus.build(use_graph=True, use_alias=True, llm=llm)
    qs = data.load_evermem_questions(topic)[:2]
    for method in ["bm25", "hybrid", "camel"]:
        for q in qs:
            q_meta = {"reference": q.get("reference"), "topic_id": topic,
                      "category": q["category"], "dimension": q["dimension"]}
            ctx = corpus.answer_context(method, q["question"], q_meta=q_meta)
            ans = answerer.answer_evermem(q["question"], ctx, q.get("options"))
            verdict = judge.judge_evermem(q["question"], q["answer"], ans)
            print(f"[{method}] {q['id']} ({q['category']}) -> {ans!r} "
                  f"verdict={parse_evermem_verdict(verdict)}")
    domain = "Finance"
    messages, _ = data.load_groupmem_domain(domain)
    corpus = Corpus(messages, corpus_key=f"groupmem_{domain}")
    corpus.build(use_graph=True, use_alias=True, llm=llm)
    qs = data.load_groupmem_questions(domain, qtype="multi_hop")[:2]
    for method in ["bm25", "camel"]:
        for q in qs:
            ctx = corpus.answer_context(method, q["question"],
                                        asker=q["asking_user_id"],
                                        q_meta={"qtype": q["qtype"]})
            ans = answerer.answer_groupmem(q["question"], ctx,
                                           asker=q["asking_user_id"])
            final = extract_final(ans)
            verdict = judge.judge_groupmem(q["question"], q["answer"], final)
            print(f"[{method}] {q['id']} ({q['qtype']}) -> {final!r} "
                  f"verdict={parse_groupmem_verdict(verdict)}")


# ---------------------------------------------------------------------------
def _preflight():
    """One real LLM round-trip before any indexing work.

    Catches a wrong/expired key or an unreachable endpoint in ~a second,
    instead of after the corpus is built and every question has exhausted its
    retries.
    """
    from .llm import LLMClient
    print(f"[config] {config.CONFIG_PATH}", flush=True)
    print(f"[config] {config.describe()}", flush=True)
    print(f"[config] code version: {CODE_VERSION}", flush=True)
    # Report the device actually chosen. The cross-encoder is ~90% of runtime,
    # so silently landing on CPU turns a 12-hour run into a multi-day one.
    try:
        from .retrieval import _resolve_device
        dev = _resolve_device()
        note = ""
        if config.RERANKER_PATH and dev == "cpu":
            note = ("  <-- reranking on CPU is ~6x slower; set embedding.device: "
                    "auto/cuda in config.yaml if a GPU is free")
        print(f"[config] embed/rerank device: {dev}{note}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        LLMClient(model=config.ANSWER_MODEL).complete(
            "Reply with OK.", "ping", use_cache=False)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\nLLM preflight failed.\n"
            f"  api      : {config.LLM_API}\n"
            f"  base_url : {config.LLM_BASE_URL}\n"
            f"  model    : {config.ANSWER_MODEL}\n"
            f"  error    : {e}\n\n"
            f"Check the `llm:` section of {config.CONFIG_PATH}.\n"
            f"For https://api.deepseek.com/v1 use `api: openai`; for the\n"
            f"/anthropic endpoint use `api: anthropic`.\n")
    print("[preflight] LLM OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("groupmem")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--qtypes", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="max questions per domain (stratified by qtype)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("evermem")
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="max questions per topic (stratified by category)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("socialmem")
    p.add_argument("--networks", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="max questions per network (stratified by qtype)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("sweep")
    p.add_argument("--bench", choices=["groupmem", "evermem"], required=True)
    p.add_argument("--param", required=True)
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--methods", nargs="*", default=["camel"])
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--run-id", default="sweep")

    p = sub.add_parser("aggregate")
    p.add_argument("--run-id", default=None)

    sub.add_parser("smoke")

    args = ap.parse_args()
    config.ensure_dirs()
    if args.cmd in ("groupmem", "evermem", "socialmem", "sweep", "smoke"):
        config.check_api_key()
        _preflight()
    if args.cmd == "groupmem":
        try:
            run_groupmem(args.domains, args.methods, args.qtypes, args.top_k,
                         args.workers, args.limit, args.run_id, args.seed)
        except _RunAborted as e:
            print(f"\n[aborted] {e}", flush=True)
        aggregate(args.run_id)
    elif args.cmd == "evermem":
        try:
            run_evermem(args.topics, args.methods, args.top_k, args.workers,
                        args.limit, args.run_id, args.seed)
        except _RunAborted as e:
            print(f"\n[aborted] {e}", flush=True)
        aggregate(args.run_id)
    elif args.cmd == "socialmem":
        try:
            run_socialmem(args.networks, args.methods, args.top_k,
                          args.workers, args.limit, args.run_id, args.seed)
        except _RunAborted as e:
            print(f"\n[aborted] {e}", flush=True)
        aggregate(args.run_id)
    elif args.cmd == "sweep":
        kw = {"limit": args.limit, "workers": args.workers}
        if args.bench == "groupmem":
            kw["domains"] = args.domains
        else:
            kw["topics"] = args.topics
        sweep(args.bench, args.param, args.values, args.methods, args.run_id, **kw)
    elif args.cmd == "aggregate":
        aggregate(args.run_id)
    elif args.cmd == "smoke":
        smoke()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
