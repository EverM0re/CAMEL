"""Central configuration for CAMEL.

Settings are resolved with the precedence:

    environment variable  >  config.yaml  >  built-in default

config.yaml is the place to put your API key, model names and paths. The
environment layer is kept because the hyper-parameter sweep uses it to vary one
setting per subprocess without rewriting the file.

Point at a different file with CAMEL_CONFIG=/path/to/other.yaml.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# config.yaml ships inside the package, so the whole thing is one uploadable
# directory. A config.yaml placed next to the package still wins, which keeps
# a machine-specific override possible without editing the tracked file.
_DEFAULT_YAML = _HERE / "config.yaml"
_PARENT_YAML = _HERE.parent / "config.yaml"


def _resolve_config_path():
    """CAMEL_CONFIG  >  ../config.yaml (local override)  >  packaged one."""
    env = os.environ.get("CAMEL_CONFIG")
    if env:
        return Path(env)
    if _PARENT_YAML.exists():
        return _PARENT_YAML
    return _DEFAULT_YAML


def _load_yaml():
    path = _resolve_config_path()
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[config] pyyaml not installed; ignoring {path}. "
              "Install it with: pip install pyyaml")
        return {}
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[config] could not parse {path}: {e}")


_CFG = _load_yaml()
CONFIG_PATH = _resolve_config_path()


def _get(section, key, default=None):
    """Read `section.key` from the YAML, falling back to `default`."""
    sec = _CFG.get(section)
    if not isinstance(sec, dict):
        return default
    v = sec.get(key, default)
    return default if v is None else v


def _s(env, section, key, default=""):
    """String setting: env var wins, then YAML, then default."""
    v = os.environ.get(env)
    if v is not None and v != "":
        return v
    return str(_get(section, key, default))


def _i(env, section, key, default):
    v = os.environ.get(env)
    if v is not None and v != "":
        return int(v)
    return int(_get(section, key, default))


def _f(env, section, key, default):
    v = os.environ.get(env)
    if v is not None and v != "":
        return float(v)
    return float(_get(section, key, default))


def _b(env, section, key, default):
    v = os.environ.get(env)
    if v is None or v == "":
        return bool(_get(section, key, default))
    return v.strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(_s("CAMEL_ROOT", "paths", "root",
                "./data"))

GROUP_DIR = ROOT / "GroupMemBench"
GROUP_DATA = GROUP_DIR / "data" / "final"
GROUP_QUESTIONS = GROUP_DIR / "questions"

EVER_DIR = ROOT / "EverMemBench"
EVER_DATA = EVER_DIR / "dataset_download"  # HF snapshot
EVER_TOPICS = ["01", "02", "03", "04", "05"]  # primary 2400-QA benchmark
EVER_BATCHES = ["004", "005", "010", "011", "016"]  # English eval batches

# SocialMemBench: a local dir holding the four parquet files, or blank to
# stream from the HuggingFace Hub (anon4data/socialmembench, CC BY 4.0).
SOCIAL_DATA = Path(_s("CAMEL_SOCIAL_DATA", "paths", "social_data", "")
                   or str(ROOT / "SocialMemBench"))

RESULTS_DIR = Path(_s("CAMEL_RESULTS_DIR", "paths", "results_dir", "")
                   or str(ROOT / "results"))
CACHE_DIR = Path(_s("CAMEL_CACHE_DIR", "paths", "cache_dir", "")
                 or str(ROOT / "cache"))
EMBED_DIR = CACHE_DIR / "embeddings"

BGE_MODEL_PATH = _s("CAMEL_BGE_PATH", "paths", "bge_model",
                    "BAAI/bge-m3")

# Optional cross-encoder reranker. Blank -> reranking degrades to fusion order.
RERANKER_PATH = _s("CAMEL_RERANKER_PATH", "paths", "reranker", "")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# "openai"    -> OpenAI-compatible  (https://api.deepseek.com/v1)
# "anthropic" -> Anthropic-compatible (https://api.deepseek.com/anthropic)
LLM_API = _s("CAMEL_LLM_API", "llm", "api", "openai").strip().lower()
LLM_BASE_URL = _s("ANTHROPIC_BASE_URL", "llm", "base_url",
                  "https://api.deepseek.com/v1")
# The SDK appends the route itself, so a base_url that already ends in the
# route would POST to ".../chat/completions/chat/completions" and 404 on every
# call. Strip it rather than fail cryptically -- it is an easy mistake to make.
for _suffix in ("/chat/completions", "/messages", "/completions"):
    if LLM_BASE_URL.rstrip("/").endswith(_suffix):
        LLM_BASE_URL = LLM_BASE_URL.rstrip("/")[: -len(_suffix)]
        print(f"[config] trimmed '{_suffix}' from base_url -> {LLM_BASE_URL}")
        break
LLM_BASE_URL = LLM_BASE_URL.rstrip("/")
LLM_API_KEY = (os.environ.get("ANTHROPIC_AUTH_TOKEN")
               or os.environ.get("ANTHROPIC_API_KEY")
               or os.environ.get("DEEPSEEK_API_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or str(_get("llm", "api_key", "") or ""))
ANSWER_MODEL = _s("CAMEL_ANSWER_MODEL", "llm", "answer_model",
                  "deepseek-chat")
JUDGE_MODEL = _s("CAMEL_JUDGE_MODEL", "llm", "judge_model", "deepseek-chat")
# The judge's own endpoint. Normally the same as the answer model's, but the
# backbone-transfer experiment repoints the ANSWER model at another server
# while the judge must stay where its model actually lives -- otherwise the
# judge's model name is sent to the answer server and 404s.
JUDGE_BASE_URL = _s("CAMEL_JUDGE_BASE_URL", "llm", "judge_base_url",
                    "").rstrip("/") or LLM_BASE_URL
JUDGE_API_KEY = (os.environ.get("CAMEL_JUDGE_API_KEY")
                 or str(_get("llm", "judge_api_key", "") or "")
                 or LLM_API_KEY)
JUDGE_API = _s("CAMEL_JUDGE_API", "llm", "judge_api",
               "").strip().lower() or LLM_API
LLM_MAX_RETRIES = _i("CAMEL_LLM_RETRIES", "llm", "max_retries", 6)


def backbones():
    """Alternative answer models for the backbone-transfer experiment.

    Returns [{name, label, base_url, api_key, api}] for every filled-in slot
    of the top-level `backbones:` list in config.yaml, with blank endpoint
    fields inherited from the `llm:` block. Slots with no model name are
    skipped, so the default config yields an empty list.
    """
    raw = _CFG.get("backbones") or []
    if not isinstance(raw, list):
        return []
    out = []
    for b in raw:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            "label": str(b.get("label") or "").strip() or name,
            "base_url": str(b.get("base_url") or "").strip() or LLM_BASE_URL,
            "api_key": str(b.get("api_key") or "").strip() or LLM_API_KEY,
            "api": str(b.get("api") or "").strip() or LLM_API,
        })
    return out
# Minimum token budget for ANY call. A reasoning model spends hidden thinking
# tokens out of max_tokens before emitting a single visible character, so a
# small budget yields an empty string -- which previously scored every
# multiple-choice category at exactly 0.0%. Raising the floor costs nothing:
# billing is on tokens actually produced, not on the cap.
LLM_MIN_TOKENS = _i("CAMEL_LLM_MIN_TOKENS", "llm", "min_tokens", 2048)
LLM_TIMEOUT = _f("CAMEL_LLM_TIMEOUT", "llm", "timeout", 120)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_DEVICE = _s("CAMEL_EMBED_DEVICE", "embedding", "device", "cpu")
EMBED_BATCH_SIZE = _i("CAMEL_EMBED_BATCH_SIZE", "embedding", "batch_size", 256)

# ---------------------------------------------------------------------------
# Retrieval (all sweepable from the experiment script)
# ---------------------------------------------------------------------------
TOP_K = _i("CAMEL_TOP_K", "retrieval", "top_k", 50)
CANDIDATE_K = _i("CAMEL_CANDIDATE_K", "retrieval", "candidate_k", 200)
MAX_CONTEXT_CHARS = _i("CAMEL_MAX_CONTEXT_CHARS", "retrieval",
                       "max_context_chars", 32000)
RRF_K = _i("CAMEL_RRF_K", "retrieval", "rrf_k", 60)
# Optional cap on the window as a fraction of the corpus. DISABLED by default
# (1.0 = no cap): it was introduced on the hypothesis that CAMEL's loss on
# small SocialMemBench networks was context dilution, but a controlled re-run
# refuted that -- capping cut the window 50 -> 11.6, recall 82.1% -> 64.8% and
# accuracy 73.8 -> 70.1. The wider window helps even on a 171-turn corpus.
TOP_K_CORPUS_FRAC = _f("CAMEL_TOP_K_CORPUS_FRAC", "retrieval",
                       "top_k_corpus_frac", 1.0)
TOP_K_MIN = _i("CAMEL_TOP_K_MIN", "retrieval", "top_k_min", 10)

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
GRAPH_HOPS = _i("CAMEL_GRAPH_HOPS", "graph", "hops", 2)
# How many top seeds seed the graph walk. Anchoring on only the top few
# misses chains whose head is outranked by lexical near-duplicates.
GRAPH_ANCHORS = _i("CAMEL_GRAPH_ANCHORS", "graph", "anchors", 40)
# Fraction of the final budget that graph/alias expansion may occupy. The rest
# is always reserved for relevance-ranked seeds.
GRAPH_BUDGET_FRAC = _f("CAMEL_GRAPH_BUDGET_FRAC", "graph", "budget_frac", 0.10)
# Cross-role synonyms ("finance workflows" <-> "Finance Ops") are paraphrases,
# not lexical near-duplicates, so they sit well below the 0.86 used initially.
# At 0.86 the mechanism was completely inert: -noterm was byte-identical to the
# full system on all 90 GroupMemBench questions. The sweep tunes this.
ALIAS_SIM = _f("CAMEL_ALIAS_SIM", "graph", "alias_sim", 0.78)

# --- Contextual Term Grounding (CTG) --------------------------------------
# Terminology alignment is posed as typed relation inference, not thresholded
# similarity. `alias_sim` above is demoted to the cosine BASELINE's threshold;
# the candidate stage below uses top-k recall with only a permissive floor.
#   term_backend: "nli" | "reranker" | "cosine"
TERM_BACKEND = _s("CAMEL_TERM_BACKEND", "term_align", "backend", "reranker")
TERM_BACKEND_FALLBACK = _s("CAMEL_TERM_FALLBACK", "term_align", "fallback",
                           "reranker")
TERM_NLI_PATH = _s("CAMEL_TERM_NLI_PATH", "term_align", "nli_model", "")
TERM_TOPK = _i("CAMEL_TERM_TOPK", "term_align", "top_k", 5)
# Entailment probability required in BOTH directions for `equivalent`.
TERM_TAU = _f("CAMEL_TERM_TAU", "term_align", "tau", 0.5)
# Permissive recall floor for candidate generation (NOT an alias decision).
TERM_CAND_FLOOR = _f("CAMEL_TERM_CAND_FLOOR", "term_align", "cand_floor", 0.55)
# Fraction of tau above which a non-entailing pair still counts as `related`.
TERM_RELATED_FRAC = _f("CAMEL_TERM_RELATED_FRAC", "term_align",
                       "related_frac", 0.7)
# Calibrate tau to the observed score distribution instead of trusting the
# backend to emit probabilities. Without this a relevance cross-encoder
# labelled 100% of candidate pairs `equivalent`.
TERM_CALIBRATE = _b("CAMEL_TERM_CALIBRATE", "term_align", "calibrate", True)
# Quantile of observed directional scores used as the entailment threshold.
# 0.85 keeps `equivalent` a minority label, as co-reference should be.
TERM_QUANTILE = _f("CAMEL_TERM_QUANTILE", "term_align", "quantile", 0.85)

# ---------------------------------------------------------------------------
# Feature switches (ablations flip these)
# ---------------------------------------------------------------------------
USE_RERANK = _b("CAMEL_USE_RERANK", "features", "rerank", True)
# How many candidates the cross-encoder actually scores. Reranking the full
# candidate pool on CPU took ~30s per question (97% of total runtime); the
# tail of the pool almost never reaches the final top_k, so scoring a shallow
# prefix costs little accuracy and is several times faster.
RERANK_DEPTH = _i("CAMEL_RERANK_DEPTH", "features", "rerank_depth", 48)
DECOMPOSE = _b("CAMEL_DECOMPOSE", "features", "decompose", True)
USE_TIMELINE = _b("CAMEL_USE_TIMELINE", "features", "timeline", True)
MAX_SUBQUERIES = _i("CAMEL_MAX_SUBQUERIES", "features", "max_subqueries", 3)
SUBQUERY_K = _i("CAMEL_SUBQUERY_K", "features", "subquery_k", 12)
# Share of the window reserved for per-sub-query hits, so a multi-hop chain can
# actually be completed rather than one endpoint crowding out the other.
SUBQUERY_RESERVE_FRAC = _f("CAMEL_SUBQUERY_RESERVE_FRAC", "features",
                           "subquery_reserve_frac", 0.30)

# ---------------------------------------------------------------------------
# Benchmark taxonomies
# ---------------------------------------------------------------------------
GROUP_DOMAINS = ["Finance", "Technology", "Healthcare", "Manufacturing"]
GROUP_QTYPES = ["multi_hop", "knowledge_update", "temporal", "user_implicit",
                "term_ambiguity", "abstention"]

# EverMemBench category prefix -> (dimension, name)
EVER_CATEGORIES = {
    "F_SH": ("Fine-Grained Recall", "single-hop"),
    "F_MH": ("Fine-Grained Recall", "multi-hop"),
    "F_TP": ("Fine-Grained Recall", "temporal"),
    "F_HL": ("Fine-Grained Recall", "hallucination"),
    "MA_C": ("Memory Awareness", "constraint"),
    "MA_P": ("Memory Awareness", "proactivity"),
    "MA_U": ("Memory Awareness", "updating"),
    "P_Skill": ("Profile Understanding", "skill"),
    "P_Style": ("Profile Understanding", "style"),
    "P_Title": ("Profile Understanding", "title"),
}


def ensure_dirs():
    for d in (RESULTS_DIR, CACHE_DIR, EMBED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_api_key():
    """Fail fast when no API key is set.

    Without this the run indexes the whole corpus, then every question burns
    its retries with exponential back-off before surfacing the real cause --
    and a fully-failed method still writes an (empty) result file.
    """
    if LLM_API_KEY:
        return
    raise SystemExit(
        f"\nNo LLM API key found.\n\n"
        f"Put it in {CONFIG_PATH} :\n\n"
        f"  llm:\n"
        f"    api_key: \"sk-...\"\n\n"
        f"or export one of ANTHROPIC_AUTH_TOKEN / DEEPSEEK_API_KEY.\n")


def describe():
    """One-line summary of the active LLM configuration."""
    key = LLM_API_KEY
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else ("set" if key else "MISSING")
    return (f"api={LLM_API} base_url={LLM_BASE_URL} answer={ANSWER_MODEL} "
            f"judge={JUDGE_MODEL} key={masked}")
