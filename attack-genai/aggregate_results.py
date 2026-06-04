"""Back-compat shim — 'aggregate_results' moved to 'rlatk.eval.aggregate_results'.
Re-exports all names (incl. underscore) and delegates 'python attack-genai/aggregate_results.py' via runpy.
"""
from rlatk.eval.aggregate_results import *  # noqa: F401,F403
import rlatk.eval.aggregate_results as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.eval.aggregate_results", run_name="__main__", alter_sys=True)
