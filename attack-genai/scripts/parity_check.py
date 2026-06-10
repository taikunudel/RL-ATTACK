#!/usr/bin/env python
# coding: utf-8
"""Parity checks for the encoder/similarity refactor.

Run from this directory with the project env, e.g.:
    /usa/taikun/miniconda3/envs/03_transf_py311/bin/python parity_check.py

Proves the refactor is behaviour-preserving under the defaults:
  1. Encoder factory: build_attacker('bert-base-uncased', linear_head=True) gives
     the SAME logits and SAME trainable-parameter set as the old
     BertForMaskedLM/BertConfig construction + 'cls'-in-name freeze rule.
  2. USE scorer: USESimilarityScorer / build_scorer('use') reproduce the legacy
     getUSEcosSimilarity math EXACTLY (best-effort; needs the USE model).
  3. embedding_api adapter: smoke test (no network).
"""
import os
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only -> deterministic

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from similarity_scorer import USESimilarityScorer, build_scorer, DEFAULT_USE_URL
from encoders import build_attacker

SRC = [
    "The quick brown fox jumps over the lazy dog.",
    "I love programming in Python.",
    "Adversarial attacks test model robustness.",
]
ADV = [
    "The quick brown fox leaps over a lazy dog.",
    "I enjoy coding in Python.",
    "Robustness of models is tested by adversarial attacks.",
]


def legacy_getUSEcosSimilarity(srcDocs, copyDocs, embed):
    """Verbatim copy of the ORIGINAL inline implementation, for comparison."""
    USEcosinSimilarity = []
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    for src, copy in zip(srcDocs, copyDocs):
        emb1, emb2 = embed([src, copy])["outputs"]
        emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
        srcEmb = torch.unsqueeze(emb1, dim=0)
        advEmb = torch.unsqueeze(emb2, dim=0)
        es = sim_metric(srcEmb, advEmb)
        USEcosinSimilarity.append(es.item())
    return USEcosinSimilarity


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


def encoder_parity():
    section("1) ENCODER FACTORY PARITY (bert-base-uncased, linear head)")
    from transformers import BertForMaskedLM, BertConfig

    name = "bert-base-uncased"
    old_cfg = BertConfig.from_pretrained(name, output_hidden_states=True)
    old = BertForMaskedLM.from_pretrained(name, config=old_cfg).to("cpu").eval()
    new = build_attacker(name, linear_head=True, device="cpu").eval()

    ids = torch.arange(1, 13).unsqueeze(0)  # fixed deterministic input [1, 12]
    attn = torch.ones_like(ids)
    with torch.no_grad():
        lo = old(ids, attn).logits
        ln = new(ids, attn).logits

    max_abs = (lo - ln).abs().max().item()
    allclose = torch.allclose(lo, ln, atol=1e-5)

    old_trainable = {n for n, p in old.named_parameters() if "cls" in n}
    new_trainable = {n for n, p in new.named_parameters() if p.requires_grad}

    print(f"logits shape           : old {tuple(lo.shape)}  new {tuple(ln.shape)}", flush=True)
    print(f"logits max|diff|        : {max_abs:.3e}", flush=True)
    print(f"logits allclose(1e-5)   : {allclose}", flush=True)
    print(f"trainable set identical : {old_trainable == new_trainable}  (n={len(new_trainable)})", flush=True)
    ok = allclose and old_trainable == new_trainable
    print(f"==> ENCODER PARITY: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def use_parity():
    section("2) USE SIMILARITY PARITY (best-effort; needs USE model)")
    try:
        import tensorflow_hub as hub
        embed = hub.load(DEFAULT_USE_URL)
    except Exception as e:
        print(f"SKIPPED: could not load USE model ({type(e).__name__}: {e})", flush=True)
        print("  -> run in the project env with USE cached/credentialed to verify.", flush=True)
        return None
    legacy = legacy_getUSEcosSimilarity(SRC, ADV, embed)
    scorer = USESimilarityScorer(embed=embed).score(SRC, ADV)
    factory = build_scorer("use", embed=embed).score(SRC, ADV)
    print(f"legacy      : {[round(x,6) for x in legacy]}", flush=True)
    print(f"scorer      : {[round(x,6) for x in scorer]}", flush=True)
    print(f"build_scorer: {[round(x,6) for x in factory]}", flush=True)
    ok = legacy == scorer == factory
    print(f"==> USE PARITY (identical floats): {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def api_smoke():
    section("3) embedding_api ADAPTER SMOKE (no network)")

    def fake_embed(texts):
        # deterministic 3-d pseudo-embedding
        return [[float(sum(bytearray(t.encode())) % 97), float(len(t)), 1.0] for t in texts]

    sc = build_scorer("embedding_api", embed_fn=fake_embed, batch_size=2)
    sims = sc.score(["abc", "abc"], ["abc", "totally different text"])
    same = abs(sims[0] - 1.0) < 1e-6  # identical strings -> cosine 1
    print(f"sims                 : {[round(x,6) for x in sims]}", flush=True)
    print(f"identical-text cos==1: {same}", flush=True)
    print(f"==> API ADAPTER SMOKE: {'PASS' if same else 'FAIL'}", flush=True)
    return same


def ffn_smoke():
    section("4) FFN-head construction smoke (was broken before refactor)")
    try:
        m = build_attacker("bert-base-uncased", linear_head=False, device="cpu").eval()
        ids = torch.arange(1, 13).unsqueeze(0)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            out = m(ids, attn)
        ok = hasattr(out, "logits") and out.logits.shape[-1] == m.config.vocab_size
        print(f"forward returns .logits of vocab width: {ok}  shape={tuple(out.logits.shape)}", flush=True)
        print(f"==> FFN HEAD SMOKE: {'PASS' if ok else 'FAIL'}", flush=True)
        return ok
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", flush=True)
        return False


if __name__ == "__main__":
    results = {
        "encoder_parity": encoder_parity(),
        "ffn_smoke": ffn_smoke(),
        "api_smoke": api_smoke(),
        "use_parity": use_parity(),
    }
    section("SUMMARY")
    for k, v in results.items():
        print(f"  {k:16s}: {'PASS' if v is True else ('SKIP' if v is None else 'FAIL')}", flush=True)
