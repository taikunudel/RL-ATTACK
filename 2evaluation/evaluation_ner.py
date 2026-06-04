"""Back-compat shim — '2evaluation/evaluation_ner.py' moved to 'rlatk.classifier.eval.evaluation_ner'.
Re-exports all names so legacy 'from evaluation_ner import ...' keeps working;
delegates 'python 2evaluation/evaluation_ner.py' via runpy. Requires rlatk on the path (pip install -e . or PYTHONPATH=src).
"""
from rlatk.classifier.eval.evaluation_ner import *  # noqa: F401,F403
import rlatk.classifier.eval.evaluation_ner as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.eval.evaluation_ner", run_name="__main__", alter_sys=True)
