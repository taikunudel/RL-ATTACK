"""Back-compat shim — 'evaluation_attacker_genai_llama_itself_verbose' moved to 'rlatk.eval.evaluation_attacker_genai_llama_itself_verbose'.
Re-exports all names (incl. underscore) and delegates 'python attack-genai/evaluation_attacker_genai_llama_itself_verbose.py' via runpy.
"""
from rlatk.eval.evaluation_attacker_genai_llama_itself_verbose import *  # noqa: F401,F403
import rlatk.eval.evaluation_attacker_genai_llama_itself_verbose as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.eval.evaluation_attacker_genai_llama_itself_verbose", run_name="__main__", alter_sys=True)
