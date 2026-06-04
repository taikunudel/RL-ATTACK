"""Back-compat shim — 'USE_computation.py' moved to 'rlatk.classifier.use'.
Re-exports all names so legacy 'from USE_computation import ...' keeps working;
delegates 'python USE_computation.py' via runpy. Requires the rlatk package on the path
(pip install -e . , or PYTHONPATH=src).
"""
from rlatk.classifier.use import *  # noqa: F401,F403
import rlatk.classifier.use as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.use", run_name="__main__", alter_sys=True)
