#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hyper-parameter sensitivity, sampled. OUR system only.
#
#   bash camel/scripts/run_hyper.sh
#
# Sweeps the parameters that the paper's claims actually rest on:
#
#   graph_budget_frac  how much of the window structural evidence may claim.
#                      0.0 disables graph/alias expansion entirely, so this
#                      doubles as a check that the mechanism is load-bearing.
#   graph_hops         expansion depth. 1 vs 2 vs 3 tests whether multi-hop
#                      traversal helps or simply adds noise.
#   alias_sim          candidate-generation floor for terminology alignment.
# Only the two curves with a shape are sampled densely; graph_hops and
# alias_sim were flat across their full range, so extra points there add
# nothing but cost.
#
#   rerank_depth       how many candidates the cross-encoder scores; this is
#                      the dominant latency term, so it is an accuracy/cost
#                      trade-off curve rather than a tuning knob.
#
# Baselines are not re-run: a sensitivity study compares our system against
# itself. Sampled, not exhaustive -- the question is the shape of each curve.
#
# Knobs: RUN_TAG, HYP_LIMIT, HYP_TOPICS, HYP_DOMAINS, HYP_K, WORKERS
# ---------------------------------------------------------------------------
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"

RUN_TAG="${RUN_TAG:-hyper_$(date -u +%Y%m%d_%H%M%S)}"
HYP_LIMIT="${HYP_LIMIT:-120}"
HYP_TOPICS="${HYP_TOPICS:-01 02}"
HYP_DOMAINS="${HYP_DOMAINS:-Finance}"
HYP_K="${HYP_K:-5}"
WORKERS="${WORKERS:-24}"
PY="${PYTHON:-python3}"

RES="$("$PY" -c 'from camel import config; print(config.RESULTS_DIR)')"

echo "run tag : $RUN_TAG"
echo "window  : k=$HYP_K   sample: $HYP_LIMIT questions per topic/domain"

sweep_one() {   # sweep_one <param> <env var> <values...>
    local param="$1" var="$2"; shift 2
    for V in "$@"; do
        TAG="${RUN_TAG}_${param}_${V}"
        LOG="$RES/$TAG/logs"; mkdir -p "$LOG"
        echo
        echo "==== $param = $V ===="
        env "$var=$V" "$PY" -m camel.run evermem \
            --topics $HYP_TOPICS --methods CAMEL \
            --top-k "$HYP_K" --limit "$HYP_LIMIT" --workers "$WORKERS" \
            --run-id "$TAG" 2>&1 | tee "$LOG/evermem.log"
        env "$var=$V" "$PY" -m camel.run groupmem \
            --domains $HYP_DOMAINS --methods CAMEL \
            --top-k "$HYP_K" --limit "$HYP_LIMIT" --workers "$WORKERS" \
            --run-id "$TAG" 2>&1 | tee "$LOG/groupmem.log"
        "$PY" -m camel.run aggregate --run-id "$TAG" >/dev/null 2>&1
    done
}

sweep_one graph_budget_frac CAMEL_GRAPH_BUDGET_FRAC \
    0.0 0.05 0.10 0.15 0.20 0.25 0.40
sweep_one graph_hops        CAMEL_GRAPH_HOPS        1 2 3
sweep_one alias_sim         CAMEL_ALIAS_SIM         0.70 0.78 0.86
sweep_one rerank_depth      CAMEL_RERANK_DEPTH      \
    16 32 48 64 96

echo
echo "Done. Download the directories matching: $RES/${RUN_TAG}_*"
