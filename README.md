# CAMEL: Collaborative Agentic Memory via Evidence Linking across Speakers and Time

Reference implementation for the paper, released for review.

Collaborative memory — the shared, evolving record of a group conversation — is
harder than the dyadic memory most systems target. The same concept surfaces
under role-specific vocabulary, decisions are superseded rather than deleted,
and the evidence for a single question is scattered across speakers, groups and
months. CAMEL treats this as a problem of **assembling evidence** rather than
ranking passages.

## How it works

CAMEL has an offline ingestion stage that builds a heterogeneous memory graph
(no LLM calls), and an online retrieval stage with four components:

| Component | What it does |
|---|---|
| **Seeding** | Fuses BM25 and dense rankings into a wide candidate pool; decomposes compositional questions into per-anchor sub-queries |
| **Structure** | Expands along graph edges and cross-role term equivalences, admitted only under *bidirectional* entailment rather than a similarity threshold |
| **Speaker** | Injects the speaker-profile block the query type needs; promotes the asker's own messages for first-person questions |
| **Temporal** | Marks superseded statements, resolves relative dates, and presents evidence chronologically so supersession stays legible |

`camel/engine.py` is the entry point for retrieval; each component maps to a
subsection of the paper's Method section and to one row of the ablation table.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is the heaviest dependency. If the default wheel does not match your
CUDA version, install it from [pytorch.org](https://pytorch.org) first.

## Configure

All settings live in `camel/config.yaml`. At minimum, set an API key:

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # any OpenAI-compatible endpoint
  api_key: ""                             # <- yours here
  answer_model: "gpt-4o-mini"
  judge_model:  "gpt-4o-mini"             # held fixed across all experiments
```

Environment variables override the file (`OPENAI_API_KEY`, `CAMEL_*`), which is
how the sweeps vary one setting per subprocess.

The retrieval stack needs two local models, downloaded automatically by
`sentence-transformers` on first use:

- `BAAI/bge-m3` — dense embeddings
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranking and term alignment

## Get the data

```bash
python3 -m camel.scripts.download_data
```

SocialMemBench (CC BY 4.0) downloads directly. EverMemBench and GroupMemBench
are released by their own authors under their own terms, so the script prints
where to obtain them and the exact path to place them. Expected layout:

```
data/
  EverMemBench/dataset_download/<topic>/{dialogue.json,qa_<topic>.json}
  GroupMemBench/data/final/<domain>/synthetic_domain_channels_*.json
  SocialMemBench/{conversations,qa,networks,personas}.parquet
```

## Run

```bash
bash run.sh check       # verify endpoint, GPUs and model paths first
bash run.sh main        # main comparison, all methods, k=5
bash run.sh budget      # evidence-budget sweep, k = 5, 10, 20, 50
bash run.sh ablation    # component ablation
```

Each writes JSONL to a run directory under `paths.results_dir`, one line per
question, carrying the verdict, retrieval recall, chain completeness, latency
and context size. Every table in the paper is computed from these records.

Useful overrides:

```bash
RUN_TAG=my_run LIMIT=60 WORKERS=8 CAMEL_EMBED_DEVICE=cuda:0 bash run.sh main
```

Start with `LIMIT=60` to check the pipeline end to end before committing to a
full run.

## What is compared

Ten baselines, all reimplemented over the shared corpus, encoder, answer model
and judge, so differences isolate memory structure rather than backbone:

- **Retrieval** — BM25, dense (BGE-M3), GraphRAG
- **Memory systems** — Mem0, Zep, HippoRAG, RAPTOR, ReadAgent, MemGPT, flat memory
- **Upper bound** — an oracle handed the annotated gold evidence

Running the vendors' hosted services would vary the answer model per system and
confound the comparison, so we do not.

## Layout

```
camel/
  engine.py        retrieval: the four components and their ablation switches
  graph.py         offline memory graph (reply, mention, alias, decision, time)
  termalign.py     cross-role term equivalence by bidirectional entailment
  temporal.py      date parsing, supersession, working-day arithmetic
  profiles.py      speaker profile blocks
  retrieval.py     BM25, dense, RRF fusion, cross-encoder reranking
  baselines.py     the ten baselines
  qa.py            answer and judge prompts
  data.py          benchmark loaders
  run.py           evaluation driver
  config.py        settings resolution (env > config.yaml > default)
  scripts/         experiment runners and preflight
run.sh             single entry point
```

## Reproducibility notes

- Answer models are sampled, so re-running the identical question set flips a
  small fraction of individual verdicts. Paired comparisons within one run are
  unaffected; absolute levels move slightly between runs.
- The judge is held fixed across every system and backbone, so a change in
  score reflects the memory system under test rather than the grader.
- RAPTOR's Gaussian-mixture tree construction does not converge on the largest
  corpora; this is a known limitation of the method rather than of this setup,
  and the affected cells are reported as unavailable.

## License

Code released under the MIT License (see `LICENSE`). The benchmarks are
governed by their own licenses and are not redistributed here.
