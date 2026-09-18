#!/usr/bin/env python3
"""Export per-subset question counts for the three benchmarks.

    python3 -m camel.scripts.dataset_counts > DATASET_COUNTS.md

Counts come from the same loaders the experiments use, so the "Full" column
here is by construction the set a run would iterate over -- it cannot drift
from what was actually evaluated. Pass --results RUN_ID to add an "Evaluated"
column read from a recorded run's JSONL.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camel import config, data  # noqa: E402


def _evaluated(run, bench):
    """{(subset, category): n} answered in a recorded run, if one is given."""
    if not run:
        return None
    for base in (Path("results"), Path("testdata")):
        d = base / run / bench
        if d.exists():
            break
    else:
        print(f"<!-- no run {run}/{bench} -->", file=sys.stderr)
        return None
    seen = defaultdict(set)
    for f in glob.glob(str(d / "camel__*.jsonl")):
        sub = os.path.basename(f).replace(".jsonl", "").split("__")[1]
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("verdict") is not None:
                key = r.get("category") or r.get("qtype") or "?"
                seen[(sub, key)].add(r["qid"])
    return {k: len(v) for k, v in seen.items()}


# Filled in by _emit as a side effect, so the JSON export cannot drift from
# the markdown: both are produced from one pass over the same loaders.
COLLECTED = {}


def _emit(title, subsets, per_cat, ev, cat_label):
    """subsets: [(name, n)]; per_cat: {category: n}."""
    _collect(title, subsets, per_cat, ev, cat_label)
    print(f"\n## {title}\n")
    ev_tot = sum(ev.values()) if ev else None

    print(f"| {cat_label} | Full |" + (" Evaluated |" if ev else ""))
    print("|---|---:|" + ("---:|" if ev else ""))
    ev_by_cat = Counter()
    if ev:
        for (_s, c), n in ev.items():
            ev_by_cat[c] += n
    for c, n in sorted(per_cat.items(), key=lambda x: -x[1]):
        row = f"| {c} | {n:,} |"
        if ev:
            row += f" {ev_by_cat.get(c, 0):,} |"
        print(row)
    tot = sum(per_cat.values())
    row = f"| **Total** | **{tot:,}** |"
    if ev:
        row += f" **{ev_tot:,}** |"
    print(row)

    print("\n| Subset | Full |" + (" Evaluated |" if ev else ""))
    print("|---|---:|" + ("---:|" if ev else ""))
    ev_by_sub = Counter()
    if ev:
        for (s, _c), n in ev.items():
            ev_by_sub[s] += n
    for s, n in subsets:
        row = f"| {s} | {n:,} |"
        if ev:
            row += f" {ev_by_sub.get(s, 0):,} |"
        print(row)
    row = f"| **Total** ({len(subsets)} subsets) | **{tot:,}** |"
    if ev:
        row += f" **{ev_tot:,}** |"
    print(row)
    return tot, ev_tot


def _collect(title, subsets, per_cat, ev, cat_label):
    ev_by_cat, ev_by_sub = Counter(), Counter()
    if ev:
        for (sname, c), n in ev.items():
            ev_by_cat[c] += n
            ev_by_sub[sname] += n
    COLLECTED[title] = {
        "subset_unit": cat_label,
        "total_full": sum(per_cat.values()),
        "total_evaluated": sum(ev.values()) if ev else None,
        "by_category": {c: {"full": n,
                            "evaluated": ev_by_cat.get(c) if ev else None}
                        for c, n in sorted(per_cat.items(), key=lambda x: -x[1])},
        "by_subset": {sname: {"full": n,
                              "evaluated": ev_by_sub.get(sname) if ev else None}
                      for sname, n in subsets},
    }


def evermem(run):
    subsets, per_cat = [], Counter()
    for t in config.EVER_TOPICS:
        try:
            qs = data.load_evermem_questions(t)
        except Exception as e:                       # noqa: BLE001
            print(f"<!-- evermem topic {t}: {e} -->", file=sys.stderr)
            continue
        subsets.append((f"topic {t}", len(qs)))
        per_cat.update(q["category"] for q in qs)
    return _emit("EverMemBench", subsets, per_cat,
                 _evaluated(run, "evermem"), "Sub-task")


def groupmem(run):
    subsets, per_cat = [], Counter()
    for d in config.GROUP_DOMAINS:
        try:
            qs = data.load_groupmem_questions(d)
        except Exception as e:                       # noqa: BLE001
            print(f"<!-- groupmem domain {d}: {e} -->", file=sys.stderr)
            continue
        subsets.append((d, len(qs)))
        per_cat.update(q["qtype"] for q in qs)
    return _emit("GroupMemBench", subsets, per_cat,
                 _evaluated(run, "groupmem"), "Question type")


def socialmem(run):
    subsets, per_cat = [], Counter()
    try:
        nets = data.list_socialmem_networks()
    except Exception as e:                           # noqa: BLE001
        print(f"<!-- socialmem: {e} -->", file=sys.stderr)
        return 0, None
    for n in nets:
        qs = data.load_socialmem_questions(n)
        subsets.append((str(n), len(qs)))
        per_cat.update(q.get("category", "?") for q in qs)
    return _emit("SocialMemBench", subsets, per_cat,
                 _evaluated(run, "socialmem"), "Query type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="run id to read an Evaluated column from, "
                         "e.g. r13_topk_topk5")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the same counts as JSON to PATH")
    a = ap.parse_args()

    print("# Dataset question counts\n")
    print("Generated by `python3 -m camel.scripts.dataset_counts"
          + (f" --results {a.results}" if a.results else "") + "`.")
    print("\n**Full** is every question the benchmark ships, counted through "
          "the same loaders the experiments use."
          + (" **Evaluated** is what the named run actually scored, after the "
             "per-subset sampling cap used to bound API cost." if a.results
             else ""))

    totals = [("EverMemBench", *evermem(a.results)),
              ("GroupMemBench", *groupmem(a.results)),
              ("SocialMemBench", *socialmem(a.results))]

    print("\n---\n\n## Summary\n")
    has_ev = any(t[2] is not None for t in totals)
    print("| Benchmark | Full |" + (" Evaluated |" if has_ev else ""))
    print("|---|---:|" + ("---:|" if has_ev else ""))
    for name, full, ev in totals:
        row = f"| {name} | {full:,} |"
        if has_ev:
            row += f" {ev:,} |" if ev is not None else " --- |"
        print(row)
    if a.json:
        out = {"run": a.results, "benchmarks": COLLECTED,
               "total_full": sum(t[1] for t in totals),
               "total_evaluated": (sum(t[2] for t in totals if t[2] is not None)
                                   if has_ev else None)}
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n<!-- JSON written to {a.json} -->", file=sys.stderr)

    row = f"| **Total** | **{sum(t[1] for t in totals):,}** |"
    if has_ev:
        row += f" **{sum(t[2] or 0 for t in totals):,}** |"
    print(row)


if __name__ == "__main__":
    main()
