"""Back-compat shim — 'get_raw_logits' moved to 'rlatk.genai.get_raw_logits'.
Re-exports every public+underscore name so existing 'from get_raw_logits import ...'
keeps working, and delegates CLI execution via runpy so 'python attack-genai/get_raw_logits.py' still runs.
"""
from rlatk.genai.get_raw_logits import *  # noqa: F401,F403
import rlatk.genai.get_raw_logits as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.genai.get_raw_logits", run_name="__main__", alter_sys=True)
