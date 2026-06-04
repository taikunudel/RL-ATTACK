"""Back-compat shim — 'similarity_scorer' moved to 'rlatk.core.similarity_scorer'.
Re-exports every public+underscore name so existing 'from similarity_scorer import ...'
keeps working, and delegates CLI execution via runpy so 'python attack-genai/similarity_scorer.py' still runs.
"""
from rlatk.core.similarity_scorer import *  # noqa: F401,F403
import rlatk.core.similarity_scorer as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.core.similarity_scorer", run_name="__main__", alter_sys=True)
