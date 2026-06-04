"""Back-compat shim — 'utils.py' moved to 'rlatk.classifier.utils'.
Re-exports all names so legacy 'from utils import ...' keeps working;
delegates 'python utils.py' via runpy. Requires the rlatk package on the path
(pip install -e . , or PYTHONPATH=src).
"""
from rlatk.classifier.utils import *  # noqa: F401,F403
import rlatk.classifier.utils as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.utils", run_name="__main__", alter_sys=True)
