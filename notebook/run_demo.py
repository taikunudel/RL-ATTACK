#!/usr/bin/env python
"""Execute the code cells of demo_attack_qwen_colab.ipynb as a plain script.

Avoids needing jupyter/nbconvert (which won't build on this old-GCC node). Strips notebook
magics (% / !) and get_ipython lines. Set QWENDEMO_CHECK=1 to only compile (no execution).
"""
import json, os, sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_attack_qwen_colab.ipynb")

def cell_code(src):
    out = []
    for line in src:
        s = line.lstrip()
        if s.startswith("%") or s.startswith("!") or s.startswith("get_ipython"):
            continue
        out.append(line)
    return "".join(out)

nb = json.load(open(NB))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
blocks = [cell_code(c["source"]) for c in code_cells]
full = "\n\n".join(b for b in blocks if b.strip())

if os.environ.get("QWENDEMO_CHECK"):
    compile(full, NB, "exec")
    print(f"[check] compiled OK: {len(code_cells)} code cells, {full.count(chr(10))+1} lines")
    sys.exit(0)

ns = {"__name__": "__main__", "__file__": NB}
for i, b in enumerate(blocks):
    if not b.strip():
        continue
    print(f"\n{'='*70}\n# ---- code cell {i} ----\n{'='*70}", flush=True)
    exec(compile(b, f"{NB}:cell{i}", "exec"), ns)
