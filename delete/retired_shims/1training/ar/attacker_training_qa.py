"""Back-compat shim — '1training/ar/attacker_training_qa.py' moved to 'rlatk.classifier.train.attacker_training_qa'.
Re-exports all names so legacy 'from attacker_training_qa import ...' keeps working;
delegates 'python 1training/ar/attacker_training_qa.py' via runpy. Requires rlatk on the path (pip install -e . or PYTHONPATH=src).
"""
from rlatk.classifier.train.attacker_training_qa import *  # noqa: F401,F403
import rlatk.classifier.train.attacker_training_qa as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.classifier.train.attacker_training_qa", run_name="__main__", alter_sys=True)
