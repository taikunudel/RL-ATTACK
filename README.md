# rlatk

A reinforcement-learning framework that trains an **attacker** to rewrite text documents
so a *victim* model is fooled — a classifier flips its label, or a safety guard / LLM is
bypassed — while the document's meaning is preserved. The attacker proposes token-level
substitutions (local BERT); the reward combines victim-probability shift with semantic
similarity (Universal Sentence Encoder) so rewrites succeed without becoming gibberish.

## Layout (flat — one repo, attackers at the same level)

```
rlatk/
├── core/                 # rlatk.core   — shared encoders, similarity scoring, paths
├── attack-classifier/    # rlatk.classifier — NLI / QA / NER label-flip attacks
│   ├── use.py  utils.py
│   ├── train/  eval/     #   importable: rlatk.classifier.train / .eval
│   └── scripts/          #   experiment drivers (NOT imported)
├── attack-genai/         # rlatk.genai  — Llama-Guard / LLM-bypass attacks
│   ├── train.py eval.py metrics.py ...
│   ├── victims/          #   importable: rlatk.genai.victims (llama-guard, openai, gemini)
│   ├── scripts/          #   run_*.sh, train/eval/test/plot drivers, a40_wrappers/ (NOT imported)
│   └── docs/             #   A40 handover notes, model sources
├── notebook/             # the single live demo notebook + its data/launchers
│   ├── attack_demo.ipynb
│   ├── advbench_harmful_behaviors.csv
│   └── run_demo.py  run_demo.sbatch
├── pyproject.toml        # one package; maps the dirs above onto the rlatk.* namespace
├── environment.yml
└── README_legacy.md      # previous README, kept for reference
```

## Install

```bash
pip install -e .
python -c "import rlatk.core, rlatk.classifier, rlatk.genai; print('ok')"
```

The top-level dirs map onto the `rlatk.*` import namespace (see `pyproject.toml`), so every
existing `import rlatk.core / rlatk.classifier / rlatk.genai` keeps working unchanged.

## Notes on this restructure

This branch flattens the previous **parent-repo + `rl_atk` submodule** split (which pointed
the parent and submodule at the *same* GitHub repo) into one tree. Decisions made while
assembling it, for review:

- **Code only.** Run outputs (`grid_runs*/`, `eva_results/`, trained `*.pt`, `*.log`,
  `bfg.jar`, 66 GB of artifacts) were left behind and are now `.gitignore`d.
- **De-duplicated.** Flat copies in the old `attack-genai/` that duplicated the library
  (`encoders.py`, `similarity_scorer.py`, `aggregate_results.py`, `get_raw_logits.py`,
  the victim judges) were dropped in favour of the canonical `core/` and `attack-genai/`
  + `attack-genai/victims/` versions.
- **Notebook:** `demo_attack_qwen_colab.ipynb` was taken as the canonical one and renamed
  `attack_demo.ipynb`. Swap if you meant `evaluation_metrics.ipynb` / `temp.ipynb`.
- **attack-nli:** the submodule's `attack-nli/` drivers were placed under
  `attack-classifier/scripts/` (same victim family). Merge with `train/` if desired.
- **TODO at cutover:** smoke-test the editable install (above) in the real env before
  relying on it; the `pyproject.toml` packaging is untested here.
