"""Back-compat shim — '2evaluation/evaluation_nli.py' moved to 'rlatk.classifier.eval.evaluation_nli'.
Re-exports all names so legacy 'from evaluation_nli import ...' keeps working;
delegates 'python 2evaluation/evaluation_nli.py' via runpy. Requires rlatk on the path (pip install -e . or PYTHONPATH=src).
"""
from rlatk.classifier.eval.evaluation_nli import *  # noqa: F401,F403
import rlatk.classifier.eval.evaluation_nli as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.eval.evaluation_nli", run_name="__main__", alter_sys=True)
