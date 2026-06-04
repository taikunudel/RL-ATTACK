"""Back-compat shim — '2evaluation/evaluation_qa.py' moved to 'rlatk.classifier.eval.evaluation_qa'.
Re-exports all names so legacy 'from evaluation_qa import ...' keeps working;
delegates 'python 2evaluation/evaluation_qa.py' via runpy. Requires rlatk on the path (pip install -e . or PYTHONPATH=src).
"""
from rlatk.classifier.eval.evaluation_qa import *  # noqa: F401,F403
import rlatk.classifier.eval.evaluation_qa as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.eval.evaluation_qa", run_name="__main__", alter_sys=True)
