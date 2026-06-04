# Models & Sources — systematic reference

**Principle: LOCAL is always the priority** (free, no rate limits). Use a hosted API only
when no local server is available. "Local" = a vLLM OpenAI-compatible server you host
(localhost / infodeep / **infochain**) or an in-process model (transformers / TF-Hub).

Hosts: **infodeep** = `128.4.10.226` (this GPU box; `infodeep`≈`localhost`).
**infochain** = `128.4.30.138` (peer GPU box; host Llama Guard here via `host_judge_infochain.sh`).

---

## Step → role → model → source

| Pipeline step (script) | Role | Model(s) | LOCAL source (preferred) | API source (fallback, paid/limited) | How to select |
|---|---|---|---|---|---|
| Train attacker (`train_attacker_genai.py`) | Attacker (generator) | `bert-base-uncased` | transformers (in-process) | — | always local |
| ↳ reward: victim signal | Victim | Llama-Guard-3-8B | vLLM @ `--server_url` (default `localhost:8000`) | Together if URL set to it | `--server_url` |
| ↳ reward: similarity | Encoder | USE | TF-Hub (in-process) | `embedding_api` (stub) | `--sim_scorer use` |
| Attack vs guard (`evaluation_attacker_genai.py`) | Victim (attacked) | **Llama-Guard-3-8B** | vLLM @ `--server_url` (default `localhost:8000`) | Together (if URL) | `--server_url` |
| Attack vs LLM (`evaluation_attacker_genai_llama_itself.py`) | Target (attacked) | **Llama-3-8B**, **Qwen3-8B**, Llama-2-7b-chat | vLLM @ `--server_url` (key auto `EMPTY`) | OpenAI (`gpt*` in `--target_path`), Together/OpenRouter (in `--server_url`), HF router (`--use_huggingface_api`) | `--target_path`, `--server_url` |
| ↳ harmfulness judge | Judge | Llama Guard / moderation | **`infochain`** self-hosted vLLM (NEW) ; `--judge_url` local | **Together** `Llama-Guard-4-12B` (default) ; OpenAI `omni-moderation-latest` | `--judge {infochain\|llamaguard\|openai_moderation}` |
| (standalone) moderation (`gemini_moderation.py`) | Judge | Gemini `gemini-2.0-flash` | — | Google Gemini | not on default eval path |
| (standalone baseline) `llms_adversarial_documents_evaluation.py` | Target | gpt-4o, claude-3-7-sonnet, llama3-70b/8b-8192 | — | OpenAI, Anthropic, **Groq** | `enabled_models` list |
| Quality metrics (`quality_metrics.py`, item 17) | Eval-only | GPT-2-large (PPL), roberta-base-CoLA | transformers (in-process) | — | always local |

## Judge options (the only paid-by-default piece)

| `--judge` | Source | Model | Cost | Notes |
|---|---|---|---|---|
| `infochain` *(NEW)* | infochain vLLM | `--judge_model` (default `llama-guard-3`) | **FREE** | endpoint default `http://infochain:8000/v1/chat/completions`; override `--judge_url`. No API key needed. |
| `llamaguard` *(default)* | Together API | `Llama-Guard-4-12B` (or `--judge_model`) | **PAID** | unchanged; `--judge_url` can repoint it to any local vLLM. |
| `openai_moderation` | OpenAI API | `omni-moderation-latest` | API (rate-limited) | `--max_rate_limit_errors`. |

## Source catalog

- **LOCAL — vLLM** (OpenAI-compatible): targets + judge. Free, no rate limits. Start with
  `vllm_starter.py` (local) or `host_judge_infochain.sh` (infochain).
  ⚠ The vLLM `--served-model-name` MUST match the client's `--judge_model` (judge) /
  `--target_path` (target).
- **LOCAL — in-process**: transformers (`bert-base-uncased`, `gpt2-large`, `roberta-base-CoLA`),
  TF-Hub USE. Free.
- **API — Together**: `Llama-Guard-4-12B` (default judge), or a target if `--server_url` has `together`. Paid.
- **API — OpenAI**: chat (`gpt*` targets) + `omni-moderation-latest` (judge). Paid/rate-limited.
- **API — Anthropic / Groq / Gemini / HF router**: only baselines / standalone moderation. Paid/limited.

## Recommended zero-cost setup (local-first)

```bash
# Target on infodeep (this box):
#   python vllm_starter.py            # serves a target on :8000

# Judge on infochain (run there, see host_judge_infochain.sh):
#   MODEL=meta-llama/Llama-Guard-3-8B SERVED_NAME=llama-guard-3 PORT=8000 bash host_judge_infochain.sh

# Attack, judged by infochain (FREE, no rate limit), Together untouched & still available:
python evaluation_attacker_genai_llama_itself.py \
  --atker_path bert-base-uncased --atker_mode trained \
  --target_path meta-llama/Llama-3-8B --data_name harmul_strings \
  --save_to_path ./trained_attacker --server_url http://localhost:8000/v1 \
  --judge infochain --judge_url http://infochain:8000/v1/chat/completions --judge_model llama-guard-3 \
  --sim_scorer use
```
