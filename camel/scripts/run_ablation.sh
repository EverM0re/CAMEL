#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Coarse ablation: OUR system only, five variants, sampled not exhaustive.
#
#   bash camel/scripts/run_ablation.sh
#
# Reports one row per mechanism family rather than per switch:
#   CAMEL              all mechanisms on
#   -nostructure          no graph expansion, no term alignment
#   -nospeakerblk         no profile injection, no speaker grounding
#   -notemporal           no timeline arithmetic, no version resolution
#   -noranking            no cross-encoder rerank, no query decomposition
#   -base                 hybrid seeding only (every mechanism off)
#
# No baselines are run: an ablation compares our system against itself, so
# re-running baselines here would only repeat settled numbers and cost money.
#
# Knobs: RUN_TAG, ABL_LIMIT (per topic/domain), ABL_TOPICS, ABL_DOMAINS, WORKERS
# ---------------------------------------------------------------------------
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"

RUN_TAG="${RUN_TAG:-abl_$(date -u +%Y%m%d_%H%M%S)}"
ABL_LIMIT="${ABL_LIMIT:-120}"      # sampled: enough to separate the blocks
ABL_K="${ABL_K:-5}"                # same window as the main table (Table 1)
ABL_TOPICS="${ABL_TOPICS:-01 02}"
ABL_DOMAINS="${ABL_DOMAINS:-Finance}"
ABL_SOCIAL="${ABL_SOCIAL:-1}"          # 0 to skip SocialMemBench
ABL_SOCIAL_LIMIT="${ABL_SOCIAL_LIMIT:-400}"
WORKERS="${WORKERS:-24}"
PY="${PYTHON:-python3}"

# Three blocks at one level of abstraction: what enters the pool (Seeding,
# including decomposition), how it is ordered (Ranking), and what is layered on
# top of it (Structure -- graph, terms, speaker, temporal). The five-block set
# is still available via ABL_BLOCKS=5 for reproducing r14--r18.
ABL_BLOCKS="${ABL_BLOCKS:-3}"
if [ "$ABL_BLOCKS" = "5" ]; then
    DEFAULT_VARIANTS="CAMEL CAMEL-nostructure CAMEL-nospeakerblk \
CAMEL-notemporal CAMEL-noranking CAMEL-noseedblk CAMEL-base"
else
    DEFAULT_VARIANTS="CAMEL CAMEL-noseed3 CAMEL-norank3 \
CAMEL-nostruct3 CAMEL-base"
fi
VARIANTS="${VARIANTS:-$DEFAULT_VARIANTS}"

RES="$("$PY" -c 'from camel import config; print(config.RESULTS_DIR)')"
LOG="$RES/$RUN_TAG/logs"; mkdir -p "$LOG"

echo "run tag : $RUN_TAG"
echo "blocks  : $ABL_BLOCKS"
echo "variants: $VARIANTS"
echo "sample  : $ABL_LIMIT questions per topic/domain at k=$ABL_K"

"$PY" -m camel.run evermem --topics $ABL_TOPICS --methods $VARIANTS \
    --top-k "$ABL_K" --limit "$ABL_LIMIT" --workers "$WORKERS" \
    --run-id "$RUN_TAG" 2>&1 | tee "$LOG/evermem_ablation.log"

"$PY" -m camel.run groupmem --domains $ABL_DOMAINS --methods $VARIANTS \
    --top-k "$ABL_K" --limit "$ABL_LIMIT" --workers "$WORKERS" \
    --run-id "$RUN_TAG" 2>&1 | tee "$LOG/groupmem_ablation.log"

if [ "$ABL_SOCIAL" != "0" ]; then
    "$PY" -m camel.run socialmem --methods $VARIANTS \
        --top-k "$ABL_K" --limit "$ABL_SOCIAL_LIMIT" --workers "$WORKERS" \
        --run-id "$RUN_TAG" 2>&1 | tee "$LOG/socialmem_ablation.log"
fi

"$PY" -m camel.run aggregate --run-id "$RUN_TAG" 2>&1 | tee "$LOG/aggregate.log"
echo
echo "Done. Download: $RES/$RUN_TAG"
