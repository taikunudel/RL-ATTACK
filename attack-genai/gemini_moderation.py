"""Back-compat shim — 'gemini_moderation' moved to 'rlatk.victims.guards.gemini_moderation'.
Re-exports all names (incl. underscore) and delegates 'python attack-genai/gemini_moderation.py' via runpy.
"""
from rlatk.victims.guards.gemini_moderation import *  # noqa: F401,F403
import rlatk.victims.guards.gemini_moderation as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.victims.guards.gemini_moderation", run_name="__main__", alter_sys=True)
