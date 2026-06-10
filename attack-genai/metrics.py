#!/usr/bin/env python3
"""
quality_metrics.py  --  Item 17: post-hoc, EVAL-ONLY text-quality metrics.

Two local, deterministic quality metrics computed over already-generated
adversarial examples (the saved (src_doc=x, adv_doc=x', label) triples). These
are NEVER fed into the RL reward, and they make NO target/judge/API calls.

  17.1 Fluency      -- GPT-2-Large perplexity (PPL). Lower = more fluent.
                       Median per-example PPL is primary (PPL is right-skewed);
                       headline signal is dPPL = PPL(x') - PPL(x).
  17.2 Grammaticality-- CoLA acceptability (textattack/roberta-base-CoLA).
                       Acceptability rate (% with P(acceptable) >= 0.5) primary;
                       mean P(acceptable) secondary. Higher = more grammatical.

Semantic similarity is intentionally OUT OF SCOPE here (handled by gemini-embedding,
sec 16). No SBERT.

Input  : result JSON files written by evaluation_attacker_genai*.py
         (a JSON list of records, each with keys src_doc / adv_doc / ...).
Output : a tidy CSV (all metrics + CIs) and a LaTeX table (headline columns),
         aggregated per (method, target, dataset, seed) with multi-seed-ready
         confidence intervals.

Text form: by default we score the RAW stored text. The stored text is
BERT-detokenized/lowercased ("[ english ]"), which inflates absolute PPL and
CoLA noise -- but the normalization is identical for x and x', so dPPL and
delta-accept (the headline) are unaffected. Use --text-mode detok|both to also
produce a deterministically cleaned variant for readability / robustness checks.

Example:
  python quality_metrics.py --inputs "archive/eva_*llama-3-8b*.json" \
      --out-prefix out/quality --device auto
  python quality_metrics.py --sanity        # wiring self-test, no data needed
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(it, **kw):
        return it


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def pick_device(spec: str, min_free_gb: float = 4.0):
    """auto: pick the CUDA device with the most free memory (skip busy GPUs on a
    shared box); fall back to MPS then CPU. An explicit spec is honored as-is."""
    import torch
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        best, best_free = None, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except Exception:
                free = 0
            if free > best_free:
                best, best_free = i, free
        if best is not None and best_free >= min_free_gb * (1024 ** 3):
            print(f"[device] cuda:{best} ({best_free / 1024**3:.1f} GiB free)")
            return torch.device(f"cuda:{best}")
        print(f"[device] no CUDA GPU with >={min_free_gb} GiB free "
              f"(best={best_free / 1024**3:.2f} GiB) -> CPU")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Deterministic light detokenizer (applied identically to x and x')
# --------------------------------------------------------------------------- #
_CONTRACTIONS = [" n't", " 's", " 're", " 've", " 'll", " 'd", " 'm", " '"]


def detokenize(text: str) -> str:
    """Conservative, deterministic cleanup of BERT-detok artifacts. No learned
    truecasing (that could bias x vs x'). Same transform for x and x'."""
    s = text
    s = s.replace(" ##", "")                       # rejoin wordpieces if present
    for c in _CONTRACTIONS:                         # " n't" -> "n't", " 's" -> "'s"
        s = s.replace(c, c.lstrip())
    s = re.sub(r"\s+([,.!?;:%)\]\}'\"])", r"\1", s)  # drop space before closers/punct
    s = re.sub(r"([(\[\{$#@])\s+", r"\1", s)         # drop space after openers
    s = re.sub(r"\s{2,}", " ", s).strip()            # collapse runs of spaces
    s = re.sub(r"\bi\b", "I", s)                      # standalone i -> I
    # capitalize first alpha char and the first alpha after sentence punctuation
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s


def transform(text: str, mode_raw: bool) -> str:
    return text if mode_raw else detokenize(text)


# --------------------------------------------------------------------------- #
# Fluency: GPT-2-Large perplexity
# --------------------------------------------------------------------------- #
def _resolve_dtype(name: str, device):
    """Map a dtype name to a torch dtype; force fp32 on CPU (half is CPU-unfriendly)."""
    import torch
    if not name or name == "float32":
        return torch.float32
    if device.type == "cpu":
        print(f"[warn] dtype={name} requested on CPU -> using float32 (half precision is CPU-unfriendly)")
        return torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.float32)


class Fluency:
    CTX = 1024  # GPT-2 context window

    def __init__(self, model_name: str, device, dtype=None, stride: int = 512):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.torch = torch
        self.dev = device
        self.stride = stride
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.lm = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(device)

    def ppl(self, text: str) -> float:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        n = ids.size(1)
        if n < 2:
            return float("nan")  # too short to score
        if n <= self.CTX:
            with torch.no_grad():
                loss = self.lm(ids, labels=ids).loss  # mean token CE (HF shifts internally)
            return math.exp(loss.item())
        # >1024 tokens: strided sliding-window NLL, then exp(weighted mean) (HF recipe)
        nll_sum, n_tok, prev_end = 0.0, 0, 0
        for begin in range(0, n, self.stride):
            end = min(begin + self.CTX, n)
            trg_len = end - prev_end
            chunk = ids[:, begin:end]
            target = chunk.clone()
            target[:, :-trg_len] = -100
            with torch.no_grad():
                loss = self.lm(chunk, labels=target).loss  # mean CE over trg_len tokens
            nll_sum += loss.item() * trg_len
            n_tok += trg_len
            prev_end = end
            if end == n:
                break
        return math.exp(nll_sum / max(n_tok, 1))


# --------------------------------------------------------------------------- #
# Grammaticality: CoLA acceptability
# --------------------------------------------------------------------------- #
# Sentences used to EMPIRICALLY resolve which class index is "acceptable"
# (do NOT trust label index blindly; verify -- see plan caveat).
_ACC_GOOD = [
    "The committee approved the new budget after a long discussion.",
    "She quickly realized that the answer had been obvious all along.",
    "Researchers published their findings in a peer-reviewed journal.",
]
_ACC_BAD = [
    "Quickly the the realized answer obvious been all along had.",
    "Budget committee approve new long after discussion the a.",
    "Airportilis the runned over fastly without of the saying.",
]


class Grammar:
    def __init__(self, model_name: str, device, dtype=None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mod = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype).eval().to(device)
        self.acc_idx, self.resolution = self._resolve_acceptable_index()

    def _logits(self, texts):
        torch = self.torch
        enc = self.tok(list(texts), return_tensors="pt", truncation=True,
                       padding=True, max_length=self.max_length).to(self.dev)
        with torch.no_grad():
            return self.mod(**enc).logits.softmax(-1).cpu().numpy()

    def _resolve_acceptable_index(self):
        """Resolve the 'acceptable' class index. Try config.id2label by name,
        then ALWAYS validate empirically; trust the empirical result on conflict."""
        id2label = {int(k): str(v) for k, v in self.mod.config.id2label.items()}
        name_idx = None
        for idx, name in id2label.items():
            nl = name.lower()
            if nl == "acceptable" or (nl.endswith("acceptable") and "un" not in nl):
                name_idx = idx
        # empirical: which class scores higher on good vs bad sentences
        good = self._logits(_ACC_GOOD)
        bad = self._logits(_ACC_BAD)
        ncls = good.shape[1]
        margins = [good[:, c].mean() - bad[:, c].mean() for c in range(ncls)]
        emp_idx = int(np.argmax(margins))
        if name_idx is None:
            note = (f"id2label={id2label} had no 'acceptable' name; using EMPIRICAL "
                    f"index {emp_idx} (good-bad margins={np.round(margins,3).tolist()}).")
            return emp_idx, note
        if name_idx != emp_idx:
            note = (f"WARNING: name-based index {name_idx} ({id2label[name_idx]}) "
                    f"DISAGREES with empirical {emp_idx} (margins={np.round(margins,3).tolist()}). "
                    f"Trusting EMPIRICAL index {emp_idx}.")
            return emp_idx, note
        return name_idx, f"id2label={id2label}; acceptable index={name_idx} (name + empirical agree)."

    def p_acceptable(self, texts) -> np.ndarray:
        return self._logits(texts)[:, self.acc_idx]


# --------------------------------------------------------------------------- #
# Loading records + experiment metadata
# --------------------------------------------------------------------------- #
@dataclass
class FileSpec:
    path: str
    method: str
    target: str
    dataset: str
    seed: int


# eval filename grammar (tolerant): eva_{date}_{time}_{atker}_{target}_doc[_{mode}_{data}]_{masks}_{samples}.json
_KNOWN_TARGETS = ["llama-guard-3-8b", "llama-3-8b", "llama3-8b", "qwen3-8b", "qwen-3-8b"]
_KNOWN_MODES = ["trained", "untrained", "random", "train", "untrain"]
_KNOWN_DATA = ["harmul_strings", "harmful_strings", "harmful_behaviors", "wild_jailbreaking"]


def parse_filename(path: str, seed_regex: Optional[str]) -> FileSpec:
    base = os.path.basename(path)
    low = base.lower()
    target = next((t for t in _KNOWN_TARGETS if t in low), "unknown-target")
    method = next((m for m in _KNOWN_MODES if f"_{m}_" in low or f"_{m}." in low), "unknown-method")
    if method in ("train",):
        method = "trained"
    if method in ("untrain",):
        method = "untrained"
    dataset = next((d for d in _KNOWN_DATA if d in low), "unknown-data")
    seed = 42  # current code hardcodes seed=42 (no CLI flag yet)
    if seed_regex:
        m = re.search(seed_regex, base)
        if m:
            seed = int(m.group(1))
    return FileSpec(path=path, method=method, target=target, dataset=dataset, seed=seed)


def load_records(path: str, src_key: str, adv_key: str, success_key: str):
    """Return list of dicts. Robust to list-of-records and dict-of-parallel-lists."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # dict of parallel lists -> zip into records
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if not list_keys:
            raise ValueError(f"{path}: dict has no list fields to zip into records")
        nrec = min(len(data[k]) for k in list_keys)
        records = [{k: data[k][i] for k in list_keys} for i in range(nrec)]
    else:
        raise ValueError(f"{path}: unexpected top-level type {type(data)}")
    out = []
    for r in records:
        if not isinstance(r, dict) or src_key not in r or adv_key not in r:
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Aggregation with multi-seed-ready CIs
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int, level: float, rng) -> tuple:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = np.array([stat_fn(values[i]) for i in idx])
    a = (1 - level) / 2
    return (float(np.nanpercentile(boots, 100 * a)), float(np.nanpercentile(boots, 100 * (1 - a))))


def _t_ci(per_seed: list[float], level: float) -> tuple:
    arr = np.array([v for v in per_seed if not (v is None or np.isnan(v))], dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    from math import sqrt
    mean = arr.mean()
    sem = arr.std(ddof=1) / sqrt(arr.size)
    # t critical via normal approx fallback; use scipy if present
    try:
        from scipy import stats
        tcrit = stats.t.ppf(1 - (1 - level) / 2, arr.size - 1)
    except Exception:
        tcrit = 1.96
    return (mean - tcrit * sem, mean + tcrit * sem)


@dataclass
class RunStats:
    """Per-file (per-seed) scored arrays."""
    seed: int
    ppl_x: np.ndarray
    ppl_xp: np.ndarray
    acc_x: np.ndarray
    acc_xp: np.ndarray
    n: int


def aggregate_group(runs: list[RunStats], n_boot: int, level: float, rng):
    """Point estimate + CI per metric. >=2 seeds -> across-seed t-CI on per-run
    stats; 1 seed -> bootstrap-over-examples CI (clearly labeled)."""
    multi = len(runs) > 1

    def per_run(stat_fn, arr_name):
        return [stat_fn(getattr(r, arr_name)[~np.isnan(getattr(r, arr_name))])
                if getattr(r, arr_name).size else float("nan") for r in runs]

    def metric(stat_fn, arr_name):
        runvals = per_run(stat_fn, arr_name)
        point = float(np.nanmean(runvals)) if multi else runvals[0]
        if multi:
            lo, hi = _t_ci(runvals, level)
        else:
            lo, hi = _bootstrap_ci(getattr(runs[0], arr_name), stat_fn, n_boot, level, rng)
        return point, lo, hi

    med = lambda a: float(np.median(a))
    mean = lambda a: float(np.mean(a))
    rate = lambda a: float(np.mean(a >= 0.5) * 100.0)

    # dPPL handled per-example then summarized (paired x'-x)
    dppl_runs = [RunStats(r.seed, r.ppl_x, r.ppl_xp, r.acc_x, r.acc_xp, r.n) for r in runs]

    def dppl_per_run(stat_fn):
        vals = []
        for r in runs:
            d = r.ppl_xp - r.ppl_x
            d = d[~np.isnan(d)]
            vals.append(stat_fn(d) if d.size else float("nan"))
        return vals

    dppl_runvals = dppl_per_run(med)
    dppl_point = float(np.nanmean(dppl_runvals)) if multi else dppl_runvals[0]
    if multi:
        dppl_lo, dppl_hi = _t_ci(dppl_runvals, level)
    else:
        d0 = runs[0].ppl_xp - runs[0].ppl_x
        dppl_lo, dppl_hi = _bootstrap_ci(d0, med, n_boot, level, rng)

    ppl_xp_med = metric(med, "ppl_xp")
    ppl_x_med = metric(med, "ppl_x")
    ppl_xp_mean = metric(mean, "ppl_xp")
    acc_xp_rate = metric(rate, "acc_xp")
    acc_x_rate = metric(rate, "acc_x")
    acc_xp_mean = metric(mean, "acc_xp")
    acc_x_mean = metric(mean, "acc_x")

    return {
        "n_seeds": len(runs),
        "n_examples": int(sum(r.n for r in runs)),
        "ci_method": "across_seeds" if multi else "bootstrap_examples",
        "ppl_xprime_median": ppl_xp_med[0], "ppl_xprime_median_lo": ppl_xp_med[1], "ppl_xprime_median_hi": ppl_xp_med[2],
        "ppl_x_median": ppl_x_med[0],
        "ppl_xprime_mean": ppl_xp_mean[0],
        "dppl_median": dppl_point, "dppl_median_lo": dppl_lo, "dppl_median_hi": dppl_hi,
        "accept_xprime_rate": acc_xp_rate[0], "accept_xprime_rate_lo": acc_xp_rate[1], "accept_xprime_rate_hi": acc_xp_rate[2],
        "accept_x_rate": acc_x_rate[0],
        "daccept_rate": acc_xp_rate[0] - acc_x_rate[0],
        "accept_xprime_mean": acc_xp_mean[0],
        "accept_x_mean": acc_x_mean[0],
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
CSV_COLS = [
    "method", "target", "dataset", "n_seeds", "n_examples", "ci_method",
    "ppl_x_median", "ppl_xprime_median", "ppl_xprime_median_lo", "ppl_xprime_median_hi",
    "ppl_xprime_mean", "dppl_median", "dppl_median_lo", "dppl_median_hi",
    "accept_x_rate", "accept_xprime_rate", "accept_xprime_rate_lo", "accept_xprime_rate_hi",
    "daccept_rate", "accept_x_mean", "accept_xprime_mean",
]


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in CSV_COLS})


def _fmt(v, lo=None, hi=None, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:.1f}" if pct else f"{v:.1f}"
    if lo is not None and hi is not None and not (math.isnan(lo) or math.isnan(hi)):
        s += f"\\,\\scriptsize$\\pm${(hi - lo) / 2:.1f}"
    return s


def write_latex(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        r"% Auto-generated by quality_metrics.py (eval-only fluency + grammaticality)",
        r"\begin{table}[t]\centering",
        r"\caption{Text-quality of adversarial examples. PPL via GPT-2-large (median; "
        r"$\Delta$PPL$=$PPL$(x')-$PPL$(x)$, lower better $\downarrow$). "
        r"Acceptability via CoLA RoBERTa (\% with $P(\text{acceptable})\!\ge\!0.5$, higher better $\uparrow$). "
        r"$\pm$ is half the 95\% CI.}",
        r"\begin{tabular}{lll r r r r r}",
        r"\toprule",
        r"Method & Target & Data & $n$ & PPL$(x')\downarrow$ & $\Delta$PPL$\downarrow$ & Accept\%$(x')\uparrow$ & $\Delta$Accept\% \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} & {r['target']} & {r['dataset'].replace('_',' ')} & "
            f"{r['n_examples']} & "
            f"{_fmt(r['ppl_xprime_median'], r['ppl_xprime_median_lo'], r['ppl_xprime_median_hi'])} & "
            f"{_fmt(r['dppl_median'], r['dppl_median_lo'], r['dppl_median_hi'])} & "
            f"{_fmt(r['accept_xprime_rate'], r['accept_xprime_rate_lo'], r['accept_xprime_rate_hi'], pct=True)} & "
            f"{_fmt(r['daccept_rate'], pct=True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Sanity self-test (no data required)
# --------------------------------------------------------------------------- #
def run_sanity(args, device, dtype=None):
    print(f"[sanity] device = {device}  dtype = {dtype}")
    flu = Fluency(args.gpt2_model, device, dtype=dtype)
    clean = "The weather is pleasant today and the children are playing outside."
    junk = "airportilis quux the the runned over fastly without of saying glorp."
    pc, pj = flu.ppl(clean), flu.ppl(junk)
    print(f"[sanity] PPL clean={pc:.1f}  junk={pj:.1f}  -> {'OK' if pj > pc else 'FAIL: junk should be higher!'}")
    gr = Grammar(args.cola_model, device, dtype=dtype)
    print(f"[sanity] CoLA {gr.resolution}")
    ag = gr.p_acceptable([clean]); ab = gr.p_acceptable([junk])
    print(f"[sanity] P(acceptable) clean={ag[0]:.3f}  junk={ab[0]:.3f}  "
          f"-> {'OK' if ag[0] > ab[0] else 'FAIL: clean should be higher!'}")
    print(f"[sanity] short text PPL (1 token) -> {flu.ppl('the')} (expect nan)")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def score_files(specs, flu, gr, mode_raw, src_key, adv_key, success_key, only_successful, batch_size, limit):
    """Return {(method,target,dataset): [RunStats per seed]}."""
    groups: dict[tuple, list[RunStats]] = {}
    for sp in specs:
        recs = load_records(sp.path, src_key, adv_key, success_key)
        if only_successful:
            recs = [r for r in recs if r.get(success_key) in (1, True, "1")]
        if limit:
            recs = recs[:limit]
        if not recs:
            print(f"[warn] no usable records in {sp.path}")
            continue
        xs = [transform(str(r[src_key]), mode_raw) for r in recs]
        xps = [transform(str(r[adv_key]), mode_raw) for r in recs]
        ppl_x = np.array([flu.ppl(t) for t in tqdm(xs, desc=f"PPL x  {os.path.basename(sp.path)[:40]}")])
        ppl_xp = np.array([flu.ppl(t) for t in tqdm(xps, desc=f"PPL x' {os.path.basename(sp.path)[:40]}")])
        acc_x = _batched_accept(gr, xs, batch_size)
        acc_xp = _batched_accept(gr, xps, batch_size)
        rs = RunStats(sp.seed, ppl_x, ppl_xp, acc_x, acc_xp, len(recs))
        groups.setdefault((sp.method, sp.target, sp.dataset), []).append(rs)
    return groups


def _batched_accept(gr, texts, batch_size):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CoLA", leave=False):
        out.append(gr.p_acceptable(texts[i:i + batch_size]))
    return np.concatenate(out) if out else np.array([])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", help="result JSON paths / globs / dirs")
    ap.add_argument("--manifest", help="optional JSON list of {path,method,target,dataset,seed} overriding filename parsing")
    ap.add_argument("--out-prefix", default="quality_metrics_out/quality", help="output path prefix (.csv/.tex)")
    ap.add_argument("--text-mode", choices=["raw", "detok", "both"], default="raw",
                    help="raw (default; headline=delta), detok (deterministic cleanup), or both")
    ap.add_argument("--device", default="auto", help="auto|cuda|cuda:0|mps|cpu")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                    help="model precision; float16 lets gpt2-large fit a partially-free GPU (~1.6GB). CPU forces float32.")
    ap.add_argument("--gpt2-model", default="gpt2-large")
    ap.add_argument("--cola-model", default="textattack/roberta-base-CoLA")
    ap.add_argument("--src-key", default="src_doc")
    ap.add_argument("--adv-key", default="adv_doc")
    ap.add_argument("--success-key", default="adv_pred_label")
    ap.add_argument("--only-successful", action="store_true", help="restrict to successful attacks (success-key==1)")
    ap.add_argument("--seed-regex", default=r"seed[_-]?(\d+)", help="regex w/ 1 group to read seed from filename")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap records per file (debug)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--sanity", action="store_true", help="run wiring self-test and exit")
    args = ap.parse_args(argv)

    _min_free = 2.5 if args.dtype in ("float16", "bfloat16") else 4.0
    device = pick_device(args.device, min_free_gb=_min_free)
    dtype = _resolve_dtype(args.dtype, device)

    if args.sanity:
        return run_sanity(args, device, dtype)

    # resolve input file specs
    specs: list[FileSpec] = []
    if args.manifest:
        for e in json.load(open(args.manifest)):
            specs.append(FileSpec(e["path"], e.get("method", "unknown-method"),
                                  e.get("target", "unknown-target"), e.get("dataset", "unknown-data"),
                                  int(e.get("seed", 42))))
    if args.inputs:
        paths = []
        for p in args.inputs:
            if os.path.isdir(p):
                paths += sorted(glob.glob(os.path.join(p, "*.json")))
            else:
                paths += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
        manifest_paths = {s.path for s in specs}
        for p in paths:
            if p not in manifest_paths:
                specs.append(parse_filename(p, args.seed_regex))
    if not specs:
        ap.error("no input files found (use --inputs or --manifest)")

    print(f"[info] device={device}  files={len(specs)}  text-mode={args.text_mode}")
    for sp in specs:
        print(f"   - {sp.method:10s} {sp.target:16s} {sp.dataset:16s} seed={sp.seed}  {os.path.basename(sp.path)}")

    print(f"[info] loading models: fluency={args.gpt2_model}  grammar={args.cola_model}  dtype={dtype}")
    flu = Fluency(args.gpt2_model, device, dtype=dtype)
    gr = Grammar(args.cola_model, device, dtype=dtype)
    print(f"[info] CoLA {gr.resolution}")

    rng = np.random.default_rng(0)  # deterministic bootstrap
    modes = [("raw", True), ("detok", False)] if args.text_mode == "both" else \
            [(args.text_mode, args.text_mode == "raw")]

    for mode_name, mode_raw in modes:
        groups = score_files(specs, flu, gr, mode_raw, args.src_key, args.adv_key,
                             args.success_key, args.only_successful, args.batch_size, args.limit)
        rows = []
        for (method, target, dataset), runs in sorted(groups.items()):
            agg = aggregate_group(runs, args.n_boot, args.ci, rng)
            agg.update(method=method, target=target, dataset=dataset)
            rows.append(agg)
        suffix = "" if args.text_mode != "both" else f"_{mode_name}"
        csv_path = f"{args.out_prefix}{suffix}.csv"
        tex_path = f"{args.out_prefix}{suffix}.tex"
        write_csv(rows, csv_path)
        write_latex(rows, tex_path)
        print(f"\n[{mode_name}] wrote {csv_path} and {tex_path}")
        for r in rows:
            print(f"  {r['method']:10s} {r['target']:16s} {r['dataset']:16s} "
                  f"n={r['n_examples']:4d} seeds={r['n_seeds']} | "
                  f"PPL(x')={r['ppl_xprime_median']:.1f} dPPL={r['dppl_median']:+.1f} "
                  f"Accept%(x')={r['accept_xprime_rate']:.1f} dAccept%={r['daccept_rate']:+.1f} "
                  f"[CI:{r['ci_method']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
