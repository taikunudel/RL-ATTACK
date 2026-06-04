"""Back-compat shim — 'quality_metrics' moved to 'rlatk.eval.quality_metrics'.
Re-exports all names (incl. underscore) and delegates 'python attack-genai/quality_metrics.py' via runpy.
"""
from rlatk.eval.quality_metrics import *  # noqa: F401,F403
import rlatk.eval.quality_metrics as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.eval.quality_metrics", run_name="__main__", alter_sys=True)
