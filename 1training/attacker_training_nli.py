"""Back-compat shim — '1training/attacker_training_nli.py' moved to 'rlatk.classifier.train.attacker_training_nli'.
Re-exports all names so legacy 'from attacker_training_nli import ...' keeps working;
delegates 'python 1training/attacker_training_nli.py' via runpy. Requires rlatk on the path (pip install -e . or PYTHONPATH=src).
"""
from rlatk.classifier.train.attacker_training_nli import *  # noqa: F401,F403
import rlatk.classifier.train.attacker_training_nli as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.train.attacker_training_nli", run_name="__main__", alter_sys=True)
