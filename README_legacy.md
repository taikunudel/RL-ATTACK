# rlatk — RL adversarial attacks on text classifiers & GenAI safety guards

`rlatk` is a reinforcement-learning framework that trains an attacker to **rewrite text documents**
so a *victim* model is fooled — a classifier flips its label, or a safety guard / LLM is bypassed —
**while the document's meaning is preserved**.

One installable package, two attack lines + shared core:

| Subpackage | What it attacks |
|---|---|
| `rlatk.genai` | **current** — GenAI **safety guardrails** (Llama-Guard) and the LLMs they protect (Llama-3, Qwen3) |
| `rlatk.classifier` | earlier — **NLI / QA / NER** classifiers |
| `rlatk.core` | shared machinery: encoders, semantic-similarity scoring (USE), path config |

---

## Background
Safety classifiers and moderation guards (e.g. Llama-Guard) sit in front of LLMs to block harmful
requests. The robustness question this project studies: **can an adversary minimally rewrite a
harmful document so the guard marks it *safe* (or the LLM complies), without changing what the
document means?** `rlatk` answers it by *learning* such rewrites with RL, and — crucially — measures
both **attack success** and **text quality**, so attacks that succeed only by producing gibberish are
not counted as wins.

## Method
- **Attacker:** a local `bert-base-uncased` model proposes token-level substitutions to build an
  adversarial document.
- **Reward:** `victim-probability shift` + `α · semantic similarity` (Universal Sentence Encoder
  cosine) — the rewrite must both fool the victim *and* stay on-meaning. `α` (similarity weight) and
  `K` (candidate substitutions per token) are the main knobs.
- **Victims (attacked):** Llama-Guard 3 8B (safety classifier), Llama-3 8B / Qwen3-8B (LLM jailbreak).
- **Judges (score harmfulness of outputs — *not* victims):** Llama-Guard-4-12B (Together, default),
  OpenAI `omni-moderation`, Gemini `2.0-flash`.
- **Datasets:** AdvBench Harmful Strings (primary) / Harmful Behaviors.
- **Quality metrics:** GPT-2 perplexity (fluency) + CoLA (grammaticality), computed after the fact
  over saved adversarial examples.
- **Experiment driver:** a resumable, work-stealing hyperparameter grid (`α × K × seed`) with
  `.claim`/`.DONE` markers.

## Result
Results come from the grid (≈99 configs across two guards) and live under `results/` (classifier) and
`rl_atk/attack-genai/{eva_results,grid_runs}/` (genai); `analyze_hyperparams.py` +
`hyperparams_heatmap.png` summarize them.

Preliminary headline (from the hyperparameter analysis): the **`α=0.0, K=10`** attacker was the
strongest all-around *training* configuration; several "all-better" candidates were within noise, and
some ΔPPL gains are inflated by an easy baseline.

> Research in progress — treat the tables in `results/` as the source of truth, not this summary.

## Repository layout
```
rl-attack/
├── src/rlatk/
│   ├── core/         encoders, similarity_scorer (USE), paths
│   ├── genai/        train, eval, eval_llama_itself[_verbose], metrics, aggregate_results
│   │   └── victims/  llama_guard_judge, openai_moderation_judge, gemini_moderation
│   └── classifier/   train/{nli,qa…}, eval/{nli,qa,ner}, utils
├── scripts/          shell runners (train / eval / grid)
├── notebooks/        analysis notebooks
├── data/  artifacts/  results/   datasets, checkpoints, outputs — kept ON DISK, OUT of git
├── delete/           staged-for-review (never hard-deleted)
└── rl_atk/           submodule providing rlatk.core + rlatk.genai
```
Both repos contribute to one `rlatk` import namespace (PEP 420). *(Known wart: see Notes.)*

## Install
```bash
conda activate 03_transf_py311        # env with torch / transformers / tensorflow-hub / diskcache
pip install -e .                      # rlatk.classifier   (run from rl-attack/)
pip install -e rl_atk                 # rlatk.core + rlatk.genai
python -c "import rlatk.core, rlatk.genai, rlatk.classifier; print('rlatk ready')"
```

## Demo code — how to use it

**Import (works from any directory once installed):**
```python
import rlatk.genai.train                       # the GenAI attacker trainer
import rlatk.genai.eval                        # attack vs Llama-Guard
import rlatk.genai.metrics                     # PPL + CoLA quality metrics
import rlatk.classifier.eval.evaluation_nli    # NLI attack eval
```

**Run any step from the command line — each module is a runnable entry point:**
```bash
# discover the flags for a step
python -m rlatk.genai.train --help
python -m rlatk.genai.eval  --help

# train a GenAI attacker, then evaluate it vs Llama-Guard (see --help for the full flag set)
python -m rlatk.genai.train  --dataName harmul_strings   ...
python -m rlatk.genai.eval   --attackerPath <checkpoint> ...

# score quality (fluency + grammar) of saved adversarial examples
python -m rlatk.genai.metrics --inputs <eval_output.json>

# classifier line (NLI / QA / NER)
python -m rlatk.classifier.eval.evaluation_nli --help
```

**Old filenames still work** (thin compatibility shims), so existing scripts are unaffected:
```bash
python rl_atk/attack-genai/train_attacker_genai.py --help   #  ==  python -m rlatk.genai.train
```

**Run the full resumable experiment grid:**
```bash
cd rl_atk/attack-genai && bash run_grid_all.sh     # work-stealing, .claim/.DONE, resumable
```

## Configuration & secrets
API keys are read from `rl_atk/attack-genai/.env` (git-ignored) via `python-dotenv` / `os.getenv` —
**never hardcode keys in source**. Required: `OPENAI_API_KEY`, `OPENAI_MODERATION_API_KEY`,
`GOOGLE_API_KEY`, `HF_TOKEN`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY`.

## Notes
- **Hard rules:** results are **never deleted** (move to `delete/`); experiment semantics are never
  changed silently; paid APIs are gated; models run at native precision. Full dated runbook:
  `rl_atk/attack-genai/CLAUDE.md`.
- **Structural wart:** the parent repo and the `rl_atk` submodule currently point at the *same*
  GitHub repo. Long-term, collapse to a single repo (or embed a genuinely separate one).
