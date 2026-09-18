#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download models + reference baseline implementations.
#
# Run ONCE on the server before run_experiments.sh. Safe to re-run: every step
# is skipped when its target already exists.
#
#   bash scripts/setup_baselines.sh
#
# What it fetches:
#   1. bge-reranker-v2-m3   -- cross-encoder reranker (the single biggest
#                              precision win for a fixed context budget)
#   2. bge-m3               -- dense retriever (only if not already present)
#   3. Reference source for the published baselines we reimplement, so the
#      implementations can be diffed against upstream and cited precisely.
#
# The comparison itself runs our own reimplementations (camel/baselines.py)
# rather than the vendors' hosted APIs, so that every system shares the same
# corpus, embedder, answer model and judge. Calling the hosted services would
# change the answer model per system and confound the comparison.
# ---------------------------------------------------------------------------
set -uo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"    # .../camel
PROJ_DIR="$(cd "$PKG_DIR/.." && pwd)"
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CAMEL_CONFIG="${CAMEL_CONFIG:-$PKG_DIR/config.yaml}"

ROOT="${CAMEL_ROOT:-./data}"
MODEL_DIR="${CAMEL_MODEL_DIR:-./models}"
REF_DIR="$ROOT/baselines_ref"

mkdir -p "$MODEL_DIR" "$REF_DIR"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

# --- python deps -----------------------------------------------------------
log "Installing python dependencies"
pip install -q --disable-pip-version-check \
    "sentence-transformers>=3.0" "transformers>=4.40" "torch" \
    "numpy" "pyyaml" "openai" "anthropic" "huggingface_hub" 2>&1 | tail -2 || \
    warn "pip install reported problems; continuing"

# --- 1. embedding + reranker models ---------------------------------------
fetch_model() {
    local repo="$1" dest="$2"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "$repo already present at $dest -- skipping"
        return 0
    fi
    log "Downloading $repo -> $dest"
    python3 - "$repo" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
try:
    p = snapshot_download(repo_id=repo, local_dir=dest,
                          local_dir_use_symlinks=False,
                          allow_patterns=["*.json", "*.txt", "*.model",
                                          "*.safetensors", "*.bin", "*.py"])
    print("downloaded to", p)
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)
PY
}

fetch_model "BAAI/bge-reranker-v2-m3" "$MODEL_DIR/bge-reranker-v2-m3" \
    || warn "reranker download failed -- the pipeline degrades to fusion order"
fetch_model "BAAI/bge-m3" "$MODEL_DIR/bge-m3-hf" \
    || warn "bge-m3 download failed -- using the existing local snapshot"
# NLI cross-encoder for Contextual Term Grounding. ~22M-parameter backbone, so
# it is cheap; without it term alignment falls back to the bge reranker.
fetch_model "cross-encoder/nli-deberta-v3-xsmall" \
    "$MODEL_DIR/nli-deberta-v3-xsmall" \
    || warn "NLI model download failed -- term alignment will use the reranker"

# --- 1b. SocialMemBench (third benchmark; CC BY 4.0) -----------------------
SOCIAL_DIR="$ROOT/SocialMemBench"
if [ -f "$SOCIAL_DIR/qa.parquet" ]; then
    log "SocialMemBench already present at $SOCIAL_DIR -- skipping"
else
    log "Downloading SocialMemBench -> $SOCIAL_DIR"
    python3 "$PKG_DIR/scripts/fetch_socialmem.py" "$SOCIAL_DIR" \
        || warn "SocialMemBench download failed"
fi

# --- 2. reference implementations of the published baselines ---------------
# Cloned for reference/citation and to allow diffing our reimplementations
# against upstream behaviour. Nothing here is imported at run time.
clone_ref() {
    local url="$1" name="$2"
    if [ -d "$REF_DIR/$name" ]; then
        log "$name reference already cloned -- skipping"
        return 0
    fi
    log "Cloning $name reference implementation"
    git clone --depth 1 --quiet "$url" "$REF_DIR/$name" \
        || warn "could not clone $name ($url)"
}

clone_ref https://github.com/mem0ai/mem0.git                mem0
clone_ref https://github.com/OSU-NLP-Group/HippoRAG.git     hipporag
clone_ref https://github.com/parthsarthi03/raptor.git       raptor
clone_ref https://github.com/letta-ai/letta.git             memgpt
clone_ref https://github.com/getzep/graphiti.git            zep
clone_ref https://github.com/microsoft/graphrag.git         graphrag

cat > "$REF_DIR/README.md" <<'MD'
# Baseline references

These clones are for citation and behavioural reference only; the experiment
runs our own reimplementations in `camel/baselines.py`, which share the
corpus, embedder, answer model and judge with CAMEL so that the comparison
isolates memory structure rather than backbone.

| method     | system    | reference                                            |
|------------|-----------|------------------------------------------------------|
| `mem0`     | Mem0      | Chhikara et al., 2025 -- extract + ADD/UPDATE/DELETE  |
| `zep`      | Zep/Graphiti | Rasmussen et al., 2025 -- temporal KG, edge invalidation |
| `hipporag` | HippoRAG  | Gutierrez et al., NeurIPS 2024 -- personalised PageRank |
| `raptor`   | RAPTOR    | Sarthi et al., ICLR 2024 -- recursive summary tree     |
| `readagent`| ReadAgent | Lee et al., ICML 2024 -- gist memory + page lookup     |
| `memgpt`   | MemGPT/Letta | Packer et al., 2023 -- paged context + archival     |
| `graphrag` | GraphRAG  | Edge et al., 2024 -- entity-community retrieval        |
MD

# --- 3. report -------------------------------------------------------------
log "Setup summary"
RERANK_PATH="$MODEL_DIR/bge-reranker-v2-m3"
if [ -d "$RERANK_PATH" ] && [ -n "$(ls -A "$RERANK_PATH" 2>/dev/null)" ]; then
    echo "  reranker : $RERANK_PATH  (export CAMEL_RERANKER_PATH=\"$RERANK_PATH\")"
else
    echo "  reranker : NOT AVAILABLE (pipeline will use fusion order)"
fi
NLI_PATH="$MODEL_DIR/nli-deberta-v3-xsmall"
if [ -d "$NLI_PATH" ] && [ -n "$(ls -A "$NLI_PATH" 2>/dev/null)" ]; then
    echo "  NLI      : $NLI_PATH"
    echo "             set term_align.nli_model to this path and backend: nli"
else
    echo "  NLI      : not available (term alignment uses the bge reranker)"
fi
if [ -f "$SOCIAL_DIR/qa.parquet" ]; then
    echo "  social   : $SOCIAL_DIR"
else
    echo "  social   : NOT available (re-run setup, or set paths.social_data)"
fi
echo "  refs     : $REF_DIR"
echo
echo "Next:"
echo "  1. check camel/config.yaml (llm.api_key, models, paths)"
echo "  2. bash camel/scripts/run_experiments.sh"
