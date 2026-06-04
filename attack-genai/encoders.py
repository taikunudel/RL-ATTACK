"""Back-compat shim — 'encoders' moved to 'rlatk.core.encoders'.
Re-exports every public+underscore name so existing 'from encoders import ...'
keeps working, and delegates CLI execution via runpy so 'python attack-genai/encoders.py' still runs.
"""
from rlatk.core.encoders import *  # noqa: F401,F403
import rlatk.core.encoders as _impl
globals().update({k: getattr(_impl, k) for k in dir(_impl) if not k.startswith("__")})

if __name__ == "__main__":
    import runpy
    runpy.run_module("rlatk.core.encoders", run_name="__main__", alter_sys=True)
