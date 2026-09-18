#!/usr/bin/env bash
# CAMEL — one entry point for the experiments in the paper.
#
#   bash run.sh check                 environment and endpoint preflight
#   bash run.sh main                  main comparison, all methods, k=5
#   bash run.sh budget                the k sweep (k = 5, 10, 20, 50)
#   bash run.sh ablation              component ablation
#   bash run.sh all                   main + budget + ablation, in order
#
# Everything is configured in camel/config.yaml; set llm.api_key there (or
# export OPENAI_API_KEY) before the first run. Results land under
# `paths.results_dir`, one directory per RUN_TAG.
#
# Useful overrides:
#   RUN_TAG=my_run        name the output directory
#   LIMIT=60              questions per topic/domain (default: all)
#   WORKERS=8             parallel questions in flight
#   CAMEL_EMBED_DEVICE=cuda:0   pin the GPU for embeddings and reranking
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
TASK="${1:-help}"
RUN_TAG="${RUN_TAG:-camel_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-8}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

ALL_METHODS="bm25 dense graphrag mem0 zep hipporag raptor readagent memgpt \
flat_mem camel oracle"

banner() { printf '\n=== %s ===\n' "$1"; }

case "$TASK" in
  check)
    bash camel/scripts/preflight.sh
    ;;

  main)
    banner "Main comparison (k=5) -> $RUN_TAG"
    for bench in evermem groupmem socialmem; do
      "$PY" -m camel.run "$bench" --methods $ALL_METHODS \
        --top-k 5 --workers "$WORKERS" $LIMIT_ARG --run-id "$RUN_TAG"
    done
    ;;

  budget)
    banner "Evidence-budget sweep -> ${RUN_TAG}_topk*"
    RUN_TAG="$RUN_TAG" TOPK_VALUES="${TOPK_VALUES:-5 10 20 50}" \
      bash camel/scripts/run_topk.sh
    ;;

  ablation)
    banner "Component ablation -> $RUN_TAG"
    RUN_TAG="$RUN_TAG" bash camel/scripts/run_ablation.sh
    ;;

  all)
    bash "$0" main
    bash "$0" budget
    bash "$0" ablation
    ;;

  help|*)
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

printf '\nDone. Results under the run directory named above.\n'
