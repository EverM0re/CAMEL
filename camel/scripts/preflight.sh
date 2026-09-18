#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-run check: free GPUs, the configured endpoints, and the answer models.
#
#   bash camel/scripts/preflight.sh
#
# Run this BEFORE a long wave. On a shared cluster the card that was idle an
# hour ago may be full now, and a local model server that is not up fails only
# after the first batch of questions has already been paid for.
# ---------------------------------------------------------------------------
set -uo pipefail
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"
PY="${PYTHON:-python3}"

echo "=== GPUs ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | while IFS=, read -r i free tot util; do
        free=$(echo "$free" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        # 8000 MiB is the floor _best_gpu() enforces; below it we fall to CPU.
        if [ "$free" -lt 8000 ]; then
            verdict="FULL - do not use"
        elif [ "$util" -gt 50 ]; then
            verdict="busy (${util}% util) - shares compute"
        else
            verdict="usable"
        fi
        printf "  cuda:%s  %6s MiB free  %3s%% util   %s\n" \
            "$i" "$free" "$util" "$verdict"
    done
else
    echo "  nvidia-smi not found (CPU-only host?)"
fi

echo
echo "=== Endpoints ==="
"$PY" - <<'EOF'
import json, urllib.request, urllib.error
from camel import config

def probe(label, base, key, model):
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode())
        ids = [m.get("id") for m in body.get("data", [])]
        hit = "OK" if (not ids or model in ids) else "model NOT in list"
        print(f"  {label:22s} reachable   {hit}")
        if ids and model not in ids:
            print(f"      wanted {model!r}; server offers {ids[:6]}")
    except urllib.error.HTTPError as e:
        # A 401/404 still proves something is listening on that port.
        print(f"  {label:22s} HTTP {e.code} (listening, but check key/route)")
    except Exception as e:                                  # noqa: BLE001
        print(f"  {label:22s} UNREACHABLE  {type(e).__name__}: {e}")

probe("answer", config.LLM_BASE_URL, config.LLM_API_KEY, config.ANSWER_MODEL)
probe("judge", config.JUDGE_BASE_URL, config.JUDGE_API_KEY, config.JUDGE_MODEL)
for i, b in enumerate(config.backbones()):
    probe(f"backbone[{i}] {b['label'][:9]}", b["base_url"], b["api_key"],
          b["name"])
    # The judge must be reachable while THAT backbone's env is in force; this
    # is the exact pairing the backbone run uses, and the one that 404'd when
    # the answer endpoint leaked onto the judge.
    if b["base_url"] != config.JUDGE_BASE_URL:
        print(f"      judge stays on {config.JUDGE_BASE_URL} "
              f"({config.JUDGE_MODEL}) -- separate host, good")
    else:
        print(f"      judge shares this host; ensure it serves "
              f"{config.JUDGE_MODEL!r}")
EOF

echo
echo "=== Local model paths ==="
"$PY" - <<'EOF'
from pathlib import Path
from camel import config
for label, p in (("bge-m3", config.BGE_MODEL_PATH),
                 ("reranker", getattr(config, "RERANKER_PATH", ""))):
    if not p:
        print(f"  {label:10s} (unset)")
    else:
        print(f"  {label:10s} {'OK  ' if Path(p).exists() else 'MISSING '}{p}")
EOF

echo
echo "If every line above is OK, launch the wave."
