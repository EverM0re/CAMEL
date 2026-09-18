#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Backbone transfer: repeat the k=5 main comparison under alternative answer
# models, to show the findings are not an artefact of one backbone.
#
#   bash camel/scripts/run_backbone.sh
#
# Reads the `backbones:` list in camel/config.yaml -- fill in the two empty
# slots there first. One run directory is produced per backbone, tagged with
# its slot index, so results never overwrite each other.
#
# The JUDGE IS HELD FIXED across every backbone. Changing the grader at the
# same time as the system under test would confound the two effects, and the
# question here is whether OUR margin survives a different answer model, not
# whether two graders agree.
#
# Knobs: RUN_TAG, BB_K, BB_LIMIT, BB_TOPICS, BB_DOMAINS, BB_METHODS,
#        BB_SOCIAL, BB_SOCIAL_LIMIT, WORKERS
# ---------------------------------------------------------------------------
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"

RUN_TAG="${RUN_TAG:-r16}"
# Sampled, not exhaustive. The question this experiment answers is whether the
# ORDERING of methods survives a change of answer model, which a few hundred
# questions per benchmark settles; running the full sets under two more models
# would triple the API cost of the whole paper for no extra resolution. The
# sample size is recorded in the run and printed in the table caption.
BB_K="${BB_K:-5}"
BB_LIMIT="${BB_LIMIT:-60}"          # per topic/domain
BB_TOPICS="${BB_TOPICS:-01 02}"     # 2 topics  -> ~120 EverMem questions
BB_DOMAINS="${BB_DOMAINS:-Finance}" # 1 domain  -> ~60 GroupMem questions
BB_SOCIAL="${BB_SOCIAL:-1}"
BB_SOCIAL_LIMIT="${BB_SOCIAL_LIMIT:-150}"
WORKERS="${WORKERS:-24}"
PY="${PYTHON:-python3}"

# Default: the full set of methods from the main table, so Table~7 can be read
# beside Table~1 rather than as a four-row summary. Oracle is excluded (it is
# handed gold evidence, so a backbone swap tells us nothing about retrieval);
# RAPTOR is excluded on GroupMemBench by run.py already.
#
# COST WARNING. mem0, zep, raptor and readagent call the LLM at INGESTION time,
# and their caches are keyed by answer model, so a new backbone rebuilds them
# from scratch -- that is the point (reusing another model's extracted facts
# would invalidate the comparison), but it is the dominant cost of this script.
# For a cheaper run that still answers the ordering question, set
#   BB_METHODS="bm25 dense memgpt CAMEL"
# which is what the first wave used: no ingestion-time LLM calls beyond ours.
METHODS="${BB_METHODS:-bm25 dense graphrag mem0 zep hipporag raptor readagent memgpt flat_mem CAMEL}"

RES="$("$PY" -c 'from camel import config; print(config.RESULTS_DIR)')"

# Resolve the judge ONCE, from the unmodified config, and carry it into every
# backbone below. Read after the answer model has been repointed it would
# inherit the backbone's endpoint, which is the bug this guards against.
eval "$("$PY" - <<'EOF'
import shlex
from camel import config
for k, v in (("JUDGE_MODEL", config.JUDGE_MODEL),
             ("JUDGE_BASE_URL", config.JUDGE_BASE_URL),
             ("JUDGE_API_KEY", config.JUDGE_API_KEY),
             ("JUDGE_API", config.JUDGE_API)):
    print(f"{k}={shlex.quote(str(v))}")
EOF
)"

N="$("$PY" -c 'from camel import config; print(len(config.backbones()))')"
if [ "$N" = "0" ]; then
    cat <<'EOF'
No backbones configured.

Open camel/config.yaml, find the `backbones:` block and fill in at least
one slot -- the model name is the only required field:

  backbones:
    - name: "claude-sonnet-5"
      label: "Claude Sonnet 5"
    - name: "deepseek-v4-flash"
      label: "DeepSeek V4 Flash"

Leave base_url/api_key blank to reuse the endpoint in the `llm:` block above;
set them only if that backbone lives behind a different provider.
EOF
    exit 1
fi

echo "run tag   : $RUN_TAG"
echo "backbones : $N"
echo "k         : $BB_K"
echo "methods   : $METHODS"
echo "judge     : $JUDGE_MODEL @ $JUDGE_BASE_URL (held fixed)"

for I in $(seq 0 $((N - 1))); do
    # Read this slot's fields; `label` is only used for the console banner.
    eval "$("$PY" - "$I" <<'EOF'
import shlex, sys
from camel import config
b = config.backbones()[int(sys.argv[1])]
for k in ("name", "label", "base_url", "api_key", "api"):
    print(f"BB_{k.upper()}={shlex.quote(b[k])}")
EOF
)"
    TAG="${RUN_TAG}_bb${I}"
    LOG="$RES/$TAG/logs"; mkdir -p "$LOG"
    echo
    echo "==== backbone $I: $BB_LABEL ($BB_NAME) ===="

    # Only the ANSWER model changes. ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
    # are process-wide -- they move the JUDGE too, which then asks the answer
    # model's server for a model it does not host and 404s. So we pin the judge
    # to its own endpoint explicitly alongside repointing the answer model.
    export CAMEL_ANSWER_MODEL="$BB_NAME"
    export CAMEL_LLM_API="$BB_API"
    export ANTHROPIC_BASE_URL="$BB_BASE_URL"
    export ANTHROPIC_AUTH_TOKEN="$BB_API_KEY"
    export CAMEL_JUDGE_MODEL="$JUDGE_MODEL"
    export CAMEL_JUDGE_BASE_URL="$JUDGE_BASE_URL"
    export CAMEL_JUDGE_API_KEY="$JUDGE_API_KEY"
    export CAMEL_JUDGE_API="$JUDGE_API"

    "$PY" -m camel.run evermem --topics $BB_TOPICS --methods $METHODS \
        --top-k "$BB_K" --limit "$BB_LIMIT" --workers "$WORKERS" \
        --run-id "$TAG" 2>&1 | tee "$LOG/evermem.log"
    "$PY" -m camel.run groupmem --domains $BB_DOMAINS --methods $METHODS \
        --top-k "$BB_K" --limit "$BB_LIMIT" --workers "$WORKERS" \
        --run-id "$TAG" 2>&1 | tee "$LOG/groupmem.log"
    if [ "$BB_SOCIAL" != "0" ]; then
        "$PY" -m camel.run socialmem --methods $METHODS \
            --top-k "$BB_K" ${BB_SOCIAL_LIMIT:+--limit $BB_SOCIAL_LIMIT} \
            --workers "$WORKERS" --run-id "$TAG" 2>&1 \
            | tee "$LOG/socialmem.log"
    fi
    "$PY" -m camel.run aggregate --run-id "$TAG" >/dev/null 2>&1

    # Record which backbone produced this directory, so the table generator
    # can label the row without the operator having to remember the mapping.
    "$PY" - "$RES/$TAG" "$BB_NAME" "$BB_LABEL" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
(d / "backbone.json").write_text(json.dumps(
    {"name": sys.argv[2], "label": sys.argv[3]}, indent=2) + "\n")
EOF
done

echo
echo "Done. Download these directories:"
for I in $(seq 0 $((N - 1))); do echo "   $RES/${RUN_TAG}_bb${I}"; done
echo
echo "Then regenerate the table:"
echo "   CAMEL_BB_RUN=$RUN_TAG \\"
echo "     python3 paper/iclr2027/figscripts/make_backbone_table.py"
