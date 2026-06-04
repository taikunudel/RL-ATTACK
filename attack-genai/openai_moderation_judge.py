"""Back-compat shim — 'openai_moderation_judge' moved to 'rlatk.victims.guards.openai_moderation_judge'.
Re-exports all names (incl. underscore) and delegates 'python attack-genai/openai_moderation_judge.py' via runpy.
"""
from rlatk.victims.guards.openai_moderation_judge import *  # noqa: F401,F403
import rlatk.victims.guards.openai_moderation_judge as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.victims.guards.openai_moderation_judge", run_name="__main__", alter_sys=True)
