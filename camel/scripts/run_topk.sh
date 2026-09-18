#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Top-k robustness with baselines, sampled.
#
#   bash camel/scripts/run_topk.sh
#
# Runs our system AND the retained baselines at several window sizes, so the
# margin can be shown to be independent of how much context is read. Sampled
# rather than exhaustive: the question is whether the ORDERING holds across k,
# which a few hundred questions per cell already answers.
#
# Reporting (see make_paper_assets.py): per-category accuracy is emitted for
# every k, but the paper table shows only the categories whose accuracy MOVES
# across k, plus the average -- a table of nine flat columns hides the effect.
#
# Knobs: RUN_TAG, TOPK_VALUES, TOPK_LIMIT, TOPK_TOPICS, TOPK_DOMAINS, WORKERS
# ---------------------------------------------------------------------------
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"

RUN_TAG="${RUN_TAG:-topk_$(date -u +%Y%m%d_%H%M%S)}"
TOPK_VALUES="${TOPK_VALUES:-5 10 20 50}"
TOPK_LIMIT="${TOPK_LIMIT:-120}"
TOPK_TOPICS="${TOPK_TOPICS:-01 02}"
TOPK_DOMAINS="${TOPK_DOMAINS:-Finance}"
# SocialMemBench has no subset flag of its own; blank runs all 43 networks.
TOPK_SOCIAL="${TOPK_SOCIAL:-1}"        # 0 to skip SocialMemBench
TOPK_ONLY_SOCIAL="${TOPK_ONLY_SOCIAL:-0}"   # 1 to run ONLY SocialMemBench
TOPK_SOCIAL_LIMIT="${TOPK_SOCIAL_LIMIT:-}"
WORKERS="${WORKERS:-24}"
PY="${PYTHON:-python3}"

# `hybrid` is excluded from the reported baselines; BM25 and dense are the
# retrieval references, MemGPT the strongest memory system on these corpora.
METHODS="${TOPK_METHODS:-bm25 dense memgpt CAMEL}"

RES="$("$PY" -c 'from camel import config; print(config.RESULTS_DIR)')"

echo "run tag : $RUN_TAG"
echo "k values: $TOPK_VALUES"
echo "methods : $METHODS"
echo "sample  : $TOPK_LIMIT questions per topic/domain per k"

for K in $TOPK_VALUES; do
    TAG="${RUN_TAG}_topk${K}"
    LOG="$RES/$TAG/logs"; mkdir -p "$LOG"
    echo
    echo "==== k=$K ===="
    if [ "$TOPK_ONLY_SOCIAL" != "1" ]; then
    "$PY" -m camel.run evermem --topics $TOPK_TOPICS --methods $METHODS \
        --top-k "$K" --limit "$TOPK_LIMIT" --workers "$WORKERS" \
        --run-id "$TAG" 2>&1 | tee "$LOG/evermem.log"
    "$PY" -m camel.run groupmem --domains $TOPK_DOMAINS --methods $METHODS \
        --top-k "$K" --limit "$TOPK_LIMIT" --workers "$WORKERS" \
        --run-id "$TAG" 2>&1 | tee "$LOG/groupmem.log"
    fi
    if [ "$TOPK_SOCIAL" != "0" ]; then
        "$PY" -m camel.run socialmem --methods $METHODS \
            --top-k "$K" ${TOPK_SOCIAL_LIMIT:+--limit $TOPK_SOCIAL_LIMIT} \
            --workers "$WORKERS" --run-id "$TAG" 2>&1 | tee "$LOG/socialmem.log"
    fi
    "$PY" -m camel.run aggregate --run-id "$TAG" >/dev/null 2>&1
done

echo
echo "Done. Download these directories:"
for K in $TOPK_VALUES; do echo "   $RES/${RUN_TAG}_topk${K}"; done
