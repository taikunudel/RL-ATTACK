# rl-attack

RL-based **adversarial-document attacks** against NLI/QA/NER models and, in the current line of
work, against **GenAI safety guardrails** (e.g. Llama-Guard) and the LLMs they protect.

---

## ⚠️ Where the real code is

> **The actively-developed code lives in [`rl_atk/attack-genai/`](rl_atk/attack-genai/).**
> Start there. Everything outside it is older / supporting material.

`rl_atk/` is registered as a **git submodule** that points at the **same** GitHub repo
(`taikunudel/rl_atk`) on the **`ksem2026`** branch. After cloning the parent, populate it with:

```bash
git submodule update --init --recursive    # fetches rl_atk/ at the pinned commit
# or just work directly inside rl_atk/ on the ksem2026 branch
```

> **Note (structural wart, to resolve later):** the parent repo and the `rl_atk` submodule are the
> *same* GitHub repository. This self-embedding is unusual — long-term, pick one model: either the
> parent **is** `rl_atk`, or it **embeds** a genuinely separate repo. Until then, treat
> `rl_atk/attack-genai/` (branch `ksem2026`) as the source of truth.

## Runbook & living design notes

The detailed, dated runbook — pipeline, cost/API map, GPU/VRAM constraints, grid orchestration,
and **hard rules** (money gate, native precision, never-delete, don't-change-experiment-semantics)
— is in **[`rl_atk/attack-genai/CLAUDE.md`](rl_atk/attack-genai/CLAUDE.md)** (git-ignored working doc).
Read it before running GenAI experiments.

## GenAI attack pipeline (in `rl_atk/attack-genai/`)

```
train_attacker_genai.py      # train the BERT-based attacker (reward = victim prob + USE similarity)
        │
evaluation_attacker_genai.py # attack vs Llama-Guard (safety classifier)
evaluation_attacker_genai_llama_itself.py  # attack vs an LLM directly + a harmfulness judge
        │
llama_guard_judge.py / openai_moderation_judge.py / gemini_moderation.py   # pluggable judges
        │
quality_metrics.py           # eval-only fluency (GPT-2 PPL) + grammar (CoLA) over saved examples
```

Grid orchestration: `run_grid_all.sh` (work-stealing, resumable, `.claim`/`.DONE` markers) →
`run_grid_stream.sh`. Run Python in the conda env **`03_transf_py311`**.

### Secrets
API keys are loaded from `rl_atk/attack-genai/.env` (git-ignored) via `os.getenv` + `python-dotenv`.
**Never hardcode keys in source.** Required vars: `OPENAI_API_KEY`, `OPENAI_MODERATION_API_KEY`,
`GOOGLE_API_KEY`, `HF_TOKEN`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`.

## Older / supporting material (numbered pipeline)

The top-level numbered dirs are the **earlier NLI/QA/NER attack pipeline** (not the current focus):

| Dir | Stage |
|---|---|
| `0dataProcessing/` | data prep (git-ignored data) |
| `1training/` | train attackers (NLI/QA/NER) |
| `2evaluation/` | evaluation scripts + results |
| `3datasets/` | datasets (git-ignored) |

`utils.py`, `USE_computation.py` are shared helpers used by that pipeline.

## Artifacts, data & results

Model checkpoints, datasets, logs, DB caches and result dumps are **kept on disk but out of git**
(see `.gitignore`). Dated result snapshots use the `archive<date>/` / `backup<date>/` convention and
are intentionally untracked. Per project rule, results are **never deleted** — to remove something,
move it into a `delete/` folder and let the owner decide.
