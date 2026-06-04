#!/usr/bin/env python
# Llama-Guard-4-12B judge server (Flask) at NATIVE bf16, with OpenAI-style
# token logprobs so the score-based RL training loop (get_raw_logits.py) gets a
# real P_LG(y|x'), not the 0.5 fallback.
#
# Endpoints:
#   POST /v1/chat/completions {messages, logprobs?, top_logprobs?, max_tokens?}
#        -> {choices:[{message:{content}, logprobs:{content:[{token,logprob,top_logprobs:[...]}]}}]}
#   GET  /v1/models   -> {"data":[{"id": <served name>}]}   (get_raw_logits auto-detect)
#   GET  /health
#
# Env: GUARD_DTYPE (default bfloat16 = native), GUARD_PORT (default 8000),
#      GUARD_SERVED_NAME (default the HF id).
import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.cache_utils as cache_utils
import torch
import threading

# Serialize all GPU generation so concurrent requests can't race into CUDA OOM
# (the guard weights leave little headroom on a 24GB GPU). Requests queue instead.
_GEN_LOCK = threading.Lock()

# Defensive monkey-patch: handle None sliding_window in Llama 4 cache layers.
def _patch_layer(cls_name):
    cls = getattr(cache_utils, cls_name, None)
    if cls is None:
        return
    _orig = cls.__init__
    def _patched(self, sliding_window, **kwargs):
        if sliding_window is None:
            sliding_window = 2 ** 20
        _orig(self, sliding_window, **kwargs)
    cls.__init__ = _patched
for _c in ("DynamicSlidingWindowLayer", "StaticSlidingWindowLayer"):
    _patch_layer(_c)

app = Flask(__name__)

DTYPE = os.environ.get("GUARD_DTYPE", "bfloat16")
PORT = int(os.environ.get("GUARD_PORT", "8000"))
MODEL_ID = "meta-llama/Llama-Guard-4-12B"
SERVED_NAME = os.environ.get("GUARD_SERVED_NAME", MODEL_ID)
_dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]
print(f"Loading {MODEL_ID} dtype={DTYPE} port={PORT} served_as={SERVED_NAME} ...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=_dt, device_map="auto", attn_implementation="eager")
_inner = getattr(model, "model", model)
if getattr(getattr(_inner, "config", None), "attention_chunk_size", None) is None:
    try:
        _inner.config.attention_chunk_size = 8192
    except Exception:
        pass
model.generation_config.cache_implementation = None
PARAM_DTYPE = str(next(model.parameters()).dtype)
print(f"Model loaded! param_dtype={PARAM_DTYPE}", flush=True)


def _normalize(messages):
    for msg in messages:
        if isinstance(msg.get("content"), str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
    return messages


@torch.no_grad()
def _generate(messages, max_new_tokens, want_logprobs, top_k):
    inputs = tokenizer.apply_chat_template(
        _normalize(messages), return_tensors="pt", return_dict=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                      # temperature=0 / greedy -> deterministic
        return_dict_in_generate=True,
        output_scores=True,
    )
    seq = out.sequences[0]
    gen_ids = seq[input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    logprobs_content = None
    if want_logprobs:
        logprobs_content = []
        # out.scores: tuple(len = #generated steps) of [1, vocab] logits
        for step, score in enumerate(out.scores):
            if step >= gen_ids.shape[0]:
                break
            logp = torch.log_softmax(score[0].float(), dim=-1)
            tok_id = int(gen_ids[step].item())
            entry = {
                "token": tokenizer.decode([tok_id]),
                "logprob": float(logp[tok_id].item()),
            }
            if top_k and top_k > 0:
                tk = torch.topk(logp, min(top_k, logp.shape[-1]))
                entry["top_logprobs"] = [
                    {"token": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(tk.values.tolist(), tk.indices.tolist())
                ]
            logprobs_content.append(entry)
    return text, logprobs_content


MICRO_BATCH = int(os.environ.get("GUARD_MICRO_BATCH", "4"))   # fits 24GB at native bf16
MAX_PROMPT_TOKENS = int(os.environ.get("GUARD_MAX_PROMPT_TOKENS", "512"))


@torch.no_grad()
def _gen_chunk(prompts, max_new_tokens, top_k):
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=MAX_PROMPT_TOKENS, add_special_tokens=False).to(model.device)
    tokenizer.padding_side = prev_side
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, output_scores=True,
    )
    gen = out.sequences[:, input_len:]
    res = []
    for b in range(len(prompts)):
        gen_ids = gen[b]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        content = []
        for step in range(gen_ids.shape[0]):
            if step >= len(out.scores):
                break
            logp = torch.log_softmax(out.scores[step][b].float(), dim=-1)
            tok_id = int(gen_ids[step].item())
            entry = {"token": tokenizer.decode([tok_id]), "logprob": float(logp[tok_id].item())}
            if top_k and top_k > 0:
                tk = torch.topk(logp, min(top_k, logp.shape[-1]))
                entry["top_logprobs"] = [
                    {"token": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(tk.values.tolist(), tk.indices.tolist())
                ]
            content.append(entry)
        res.append({"content": text, "logprobs": {"content": content}})
    del out, enc, gen
    return res


@torch.no_grad()
def _generate_batch(list_of_messages, max_new_tokens, top_k):
    """Batched classification via MICRO-BATCHES (peak activation memory bounded so the
    12B guard fits one 24GB GPU at native bf16). LEFT padding for correct decoder-only
    batch generation. Returns list of {content, logprobs:{content:[...]}}."""
    prompts = [
        tokenizer.apply_chat_template(_normalize(m), tokenize=False, add_generation_prompt=False)
        for m in list_of_messages
    ]
    results = []
    for i in range(0, len(prompts), MICRO_BATCH):
        chunk = prompts[i:i + MICRO_BATCH]
        try:
            results.extend(_gen_chunk(chunk, max_new_tokens, top_k))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for p in chunk:                       # last-resort: one at a time
                results.extend(_gen_chunk([p], max_new_tokens, top_k))
        torch.cuda.empty_cache()
    return results


@app.route("/v1/batch_classify", methods=["POST"])
def batch_classify():
    """Body: {prompts:[str,...], max_tokens?, top_logprobs?}.
    Returns {choices:[{message:{content}, logprobs:{content:[...]}}, ...]} aligned to prompts."""
    data = request.json or {}
    prompts = data.get("prompts", [])
    max_new = int(data.get("max_tokens", 20) or 20)
    top_k = int(data.get("top_logprobs", 20) or 20)
    list_msgs = [[{"role": "user", "content": p}] for p in prompts]
    with _GEN_LOCK:                      # serialize GPU work -> no concurrent-OOM
        outs = _generate_batch(list_msgs, max_new, top_k)
    choices = [
        {"index": i, "message": {"role": "assistant", "content": o["content"]},
         "logprobs": o["logprobs"], "finish_reason": "stop"}
        for i, o in enumerate(outs)
    ]
    return jsonify({"object": "batch", "model": SERVED_NAME, "choices": choices})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "dtype": PARAM_DTYPE, "model": MODEL_ID})


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})


@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    max_new = int(data.get("max_tokens", 20) or 20)
    want_lp = bool(data.get("logprobs", False))
    top_k = int(data.get("top_logprobs", 20) or 20)
    with _GEN_LOCK:                      # serialize GPU work -> no concurrent-OOM
        text, lp_content = _generate(messages, max_new, want_lp, top_k)
    choice = {"index": 0, "message": {"role": "assistant", "content": text},
              "finish_reason": "stop"}
    if want_lp:
        choice["logprobs"] = {"content": lp_content}
    return jsonify({"object": "chat.completion", "model": SERVED_NAME, "choices": [choice]})


if __name__ == "__main__":
    print(f"Serving on 0.0.0.0:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
