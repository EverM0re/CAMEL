#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAMEL round-1 PILOT: small-scale main comparison + ablations + sweeps.
#
#   bash camel/scripts/run_experiments.sh
#
# Everything lands in  $RESULTS_DIR/<RUN_TAG>/  so successive rounds never
# overwrite each other. Download that one directory for analysis.
#
# Knobs (override on the command line, e.g. `EVER_LIMIT=120 bash ...`):
#   RUN_TAG      unique id for this round        (default: r1_<UTC timestamp>)
#   EVER_LIMIT   questions per EverMem topic     (default 120, of ~490)
#   GROUP_LIMIT  questions per GroupMem domain   (default 90,  of ~200)
#   EVER_TOPICS  topics to run                   (default "01 02")
#   GROUP_DOMAINS domains to run                 (default "Finance Technology")
#   WORKERS      parallel LLM calls              (default 16)
#   SKIP_SWEEP=1 skip the hyper-parameter sweeps
#   SKIP_LLM_BASELINES=1  skip mem0/zep/raptor/readagent/flat_mem, whose
#                         ingestion costs one LLM pass over the corpus
#   SOCIAL_METHODS / TOPK_METHODS  override the method list for those stages,
#                 e.g. SOCIAL_METHODS="camel" to re-measure only our own
#                 system after a fix, without repeating settled baselines.
#   ONLY="a b c"  run only these stages (names as printed in the banners),
#                 e.g. ONLY="groupmem_ablation" to finish a quota-killed stage.
#                 Everything already on disk is skipped automatically, so a
#                 plain re-run with the SAME RUN_TAG resumes where it stopped.
# ---------------------------------------------------------------------------
set -uo pipefail
# This script lives at <project>/camel/scripts/, so the import root -- the
# directory that CONTAINS the CAMEL package -- is two levels up.
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../camel
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"         # directory holding the package
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- configuration ---------------------------------------------------------
# Paths, models and the API key all come from config.yaml (shipped inside the
# package). Only RUN_TAG and the pilot sizes are set here; export CAMEL_* to
# override any yaml value.
CFG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"
export CAMEL_CONFIG="$CFG"
if [ ! -f "$CFG" ]; then
    echo "config.yaml not found at $CFG" >&2; exit 1
fi
# Ask the package where results go, so this script and the runner always agree.
# Plain command substitution, NOT eval: the project path may contain spaces
# ("collaboration memory"), which eval would word-split into a bogus command.
CAMEL_RESULTS_DIR="$("${PYTHON:-python3}" -c \
    'from camel import config; print(config.RESULTS_DIR)')"
if [ -z "$CAMEL_RESULTS_DIR" ]; then
    echo "could not read results_dir from $CFG" >&2; exit 1
fi

RUN_TAG="${RUN_TAG:-r1_$(date -u +%Y%m%d_%H%M%S)}"
EVER_LIMIT="${EVER_LIMIT:-120}"
GROUP_LIMIT="${GROUP_LIMIT:-90}"
EVER_TOPICS="${EVER_TOPICS:-01 02}"
GROUP_DOMAINS="${GROUP_DOMAINS:-Finance Technology}"
WORKERS="${WORKERS:-16}"
PY="${PYTHON:-python3}"

LOG_DIR="$CAMEL_RESULTS_DIR/$RUN_TAG/logs"
mkdir -p "$LOG_DIR"

banner() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }

# Stages are ordered most-important-first, and a stage that dies on an API
# quota wall must not silently burn the rest of the schedule: r6_full lost
# every stage after groupmem_main to a 403, each failing preflight in turn.
STAGES_OK=0; STAGES_FAIL=0; FAILED_STAGES=""
run() {   # run <logname> <args...>
    local name="$1"; shift
    banner "$name"
    "$PY" -m camel.run "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        STAGES_FAIL=$((STAGES_FAIL+1)); FAILED_STAGES="$FAILED_STAGES $name"
        if grep -qiE "quota|insufficient|billing|402|403" "$LOG_DIR/$name.log"; then
            echo
            echo "*** API quota exhausted during '$name'."
            echo "*** Completed stages are saved and the run is resumable:"
            echo "***   re-run the SAME command after topping up; finished"
            echo "***   questions are skipped automatically."
            AGG_ONLY=1
            return 1
        fi
        echo "[warn] stage '$name' failed (rc=$rc); continuing with the next stage"
    else
        STAGES_OK=$((STAGES_OK+1))
    fi
    return 0
}
AGG_ONLY=0
stage() {
    [ "$AGG_ONLY" = "1" ] && return 1
    if [ -n "${ONLY:-}" ]; then
        case " $ONLY " in *" $1 "*) ;; *) echo "[skip] $1 (not in ONLY)"; return 0;; esac
    fi
    run "$@"
}

# --- method sets -----------------------------------------------------------
# retrieval baselines (no LLM ingestion cost)
BASE_CHEAP="bm25 dense hybrid graphrag hipporag memgpt"
# published memory systems whose ingestion needs one LLM pass over the corpus
BASE_LLM="flat_mem mem0 zep raptor readagent"
OURS="camel"
ABLATIONS="camel-nograph CAMEL-noterm CAMEL-noversion CAMEL-nospeaker \
CAMEL-noprofile CAMEL-norerank CAMEL-nodecomp CAMEL-notimeline"
# Contextual Term Grounding ladder: cosine baseline -> context-free relation
# -> one-way entailment -> full bidirectional (== `CAMEL`).
TERM_LADDER="camel-termcosine CAMEL-termnoctx CAMEL-termoneway"

if [ "${SKIP_LLM_BASELINES:-0}" = "1" ]; then
    BASELINES="$BASE_CHEAP"
    echo "[info] skipping LLM-ingestion baselines"
else
    BASELINES="$BASE_CHEAP $BASE_LLM"
fi

echo "run tag      : $RUN_TAG"
echo "config       : $CFG"
echo "results dir  : $CAMEL_RESULTS_DIR/$RUN_TAG"
echo "ever topics  : $EVER_TOPICS  (limit $EVER_LIMIT/topic)"
echo "group domains: $GROUP_DOMAINS (limit $GROUP_LIMIT/domain)"

# ===========================================================================
# 1. MAIN COMPARISON -- ours vs. every baseline, plus the oracle upper bound
# ===========================================================================
# Order matters: if the API quota runs out mid-run, the stages that already
# finished are the ones the paper needs most. Main comparisons first (both
# benchmarks), then ablations, then the CTG ladder, then sweeps.
stage "evermem_main" evermem \
    --topics $EVER_TOPICS \
    --methods $BASELINES oracle $OURS \
    --limit "$EVER_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"

stage "groupmem_main" groupmem \
    --domains $GROUP_DOMAINS \
    --methods $BASELINES $OURS \
    --limit "$GROUP_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"

# ===========================================================================
# 1b. SOCIALMEMBENCH -- third benchmark (text-only multi-party social groups)
# ===========================================================================
if [ "${SKIP_SOCIAL:-0}" != "1" ]; then
    stage "socialmem_main" socialmem \
        ${SOCIAL_NETWORKS:+--networks $SOCIAL_NETWORKS} \
        --methods ${SOCIAL_METHODS:-$BASELINES oracle $OURS} \
        ${SOCIAL_LIMIT:+--limit $SOCIAL_LIMIT} \
        --workers "$WORKERS" --run-id "$RUN_TAG"
fi

# ===========================================================================
# 2. ABLATIONS -- one mechanism disabled at a time, same questions
# ===========================================================================
stage "evermem_ablation" evermem \
    --topics $EVER_TOPICS --methods $ABLATIONS \
    --limit "$EVER_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"

stage "groupmem_ablation" groupmem \
    --domains $GROUP_DOMAINS --methods $ABLATIONS \
    --limit "$GROUP_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"

# ===========================================================================
# 2b. TERMINOLOGY-ALIGNMENT LADDER (CTG)
# ===========================================================================
if [ "${SKIP_TERM_LADDER:-0}" != "1" ]; then
    stage "groupmem_termladder" groupmem \
        --domains $GROUP_DOMAINS --methods $TERM_LADDER \
        --limit "$GROUP_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"

    stage "evermem_termladder" evermem \
        --topics $EVER_TOPICS --methods $TERM_LADDER \
        --limit "$EVER_LIMIT" --workers "$WORKERS" --run-id "$RUN_TAG"
fi

# ===========================================================================
# 3. HYPER-PARAMETER SWEEPS -- smaller subset, one topic/domain
# ===========================================================================
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
    SWEEP_LIMIT="${SWEEP_LIMIT:-90}"
    SWEEP_TOPIC="${SWEEP_TOPIC:-01}"

    # top_k drives multi-hop recall: EverMem multi-hop items carry ~6.7 gold
    # references spread over months, and CAMEL sits 89 points below oracle
    # there, so the window size is the prime suspect.
    stage "sweep_top_k" sweep --bench evermem --param top_k \
        --values 20 30 50 80 --topics "$SWEEP_TOPIC" \
        --limit "$SWEEP_LIMIT" --workers "$WORKERS" --run-id "${RUN_TAG}_sweep"

    stage "sweep_candidate_k" sweep --bench evermem --param candidate_k \
        --values 80 120 200 --topics "$SWEEP_TOPIC" \
        --limit "$SWEEP_LIMIT" --workers "$WORKERS" --run-id "${RUN_TAG}_sweep"

    stage "sweep_rerank_depth" sweep --bench evermem --param rerank_depth \
        --values 32 48 80 --topics "$SWEEP_TOPIC" \
        --limit "$SWEEP_LIMIT" --workers "$WORKERS" --run-id "${RUN_TAG}_sweep"

    # alias_sim was inert at 0.86 (-noterm was byte-identical to the full
    # system on all 90 GroupMemBench questions); this locates a working value.
    stage "sweep_alias_sim" sweep --bench groupmem --param alias_sim \
        --values 0.70 0.78 0.86 --domains Finance \
        --limit "$SWEEP_LIMIT" --workers "$WORKERS" --run-id "${RUN_TAG}_sweep"

    stage "sweep_graph_budget" sweep --bench evermem --param graph_budget_frac \
        --values 0.0 0.10 0.25 --topics "$SWEEP_TOPIC" \
        --limit "$SWEEP_LIMIT" --workers "$WORKERS" --run-id "${RUN_TAG}_sweep"
fi

# ===========================================================================
# 3b. TOP-K COMPARISON -- ours AND the baselines at several window sizes.
# The sweep above varies top_k for `CAMEL` only; reviewers will want the
# baselines measured at the same k, since a method that merely reads more
# context should not be credited with the gain.
# ===========================================================================
if [ "${SKIP_TOPK:-0}" != "1" ]; then
    for K in ${TOPK_VALUES:-5 10 15}; do
        stage "topk${K}_evermem" evermem \
            --topics ${TOPK_TOPICS:-01 02} \
            --methods ${TOPK_METHODS:-bm25 hybrid memgpt $OURS} \
            --top-k "$K" --limit "${TOPK_LIMIT:-200}" \
            --workers "$WORKERS" --run-id "${RUN_TAG}_topk${K}"
        [ "${TOPK_SKIP_GROUPMEM:-0}" = "1" ] || \
        stage "topk${K}_groupmem" groupmem \
            --domains ${TOPK_DOMAINS:-Finance} \
            --methods ${TOPK_METHODS:-bm25 hybrid memgpt $OURS} \
            --top-k "$K" --limit "${TOPK_LIMIT:-200}" \
            --workers "$WORKERS" --run-id "${RUN_TAG}_topk${K}"
        "$PY" -m camel.run aggregate --run-id "${RUN_TAG}_topk${K}" >/dev/null 2>&1
    done
fi

# ===========================================================================
# 4. AGGREGATE + REPORT
# ===========================================================================
banner "aggregate"
"$PY" -m camel.run aggregate --run-id "$RUN_TAG" 2>&1 | tee "$LOG_DIR/aggregate.log"
"$PY" "$PKG_DIR/scripts/make_report.py" --run-id "$RUN_TAG" 2>&1 | tee "$LOG_DIR/report.log"

banner "DONE"
echo "stages ok: $STAGES_OK   failed:$([ -n "$FAILED_STAGES" ] && echo "$FAILED_STAGES" || echo " none")"
echo "Download this directory for analysis:"
echo "    $CAMEL_RESULTS_DIR/$RUN_TAG"
echo
echo "It contains:"
echo "    summary.json    per-method / per-category accuracy"
echo "    report.md       human-readable comparison + ablation deltas"
echo "    manifest.json   exact hyper-parameters used"
echo "    evermem/*.jsonl, groupmem/*.jsonl   per-question records"
echo "    logs/           stdout of every stage"
