"""Back-compat shim — 'train_attacker_genai' moved to 'rlatk.genai.train'.
Re-exports every public+underscore name so existing 'from train_attacker_genai import ...'
keeps working, and delegates CLI execution via runpy so 'python attack-genai/train_attacker_genai.py' still runs.
"""
from rlatk.genai.train import *  # noqa: F401,F403
import rlatk.genai.train as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.genai.train", run_name="__main__", alter_sys=True)
