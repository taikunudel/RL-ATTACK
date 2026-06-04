"""Stable filesystem locations for the ``rlatk`` package.

The live GenAI campaign created its ``diskcache`` in a directory that
``evaluation_attacker_genai.py`` historically hard-coded as the absolute
literal ``/usa/taikun/rl-attack/rl_atk/attack-genai``.

``DISKCACHE_DIR`` below is kept as an explicit, stable absolute path on
purpose: it must resolve to the SAME directory regardless of where this
module is imported from, so the cache hit-set is byte-identical before and
after the restructure.  Deliberately NOT ``os.path.dirname(__file__)`` —
that would change once the module is moved into ``src/rlatk/core/`` and would
silently relocate (and cold-start) the cache, changing eval results.

An optional env override (``RLATK_DISKCACHE_DIR``) lets the A40 offload point
at its bind-mounted writable dir without editing code or breaking bit-identity
— but the DEFAULT equals the old literal, so default behavior is unchanged.
"""

import os

# Canonical attack-genai dir (holds .env, cache.db, grid_runs/, eva_results/).
# Stable absolute default = the original location, so moving modules into the
# package does NOT relocate these (behavior-preserving). Override via env for A40 etc.
ATTACK_GENAI_DIR = os.environ.get(
    "RLATK_ATTACK_GENAI_DIR",
    "/usa/taikun/rl-attack/rl_atk/attack-genai",
)
DISKCACHE_DIR = os.environ.get("RLATK_DISKCACHE_DIR", ATTACK_GENAI_DIR)
GRID_RUNS_DIR = os.path.join(ATTACK_GENAI_DIR, "grid_runs")
EVA_RESULTS_DIR = os.path.join(ATTACK_GENAI_DIR, "eva_results")
ENV_FILE = os.path.join(ATTACK_GENAI_DIR, ".env")

# The historical literal, kept for an explicit equivalence assertion in tests.
_LEGACY_DISKCACHE_LITERAL = "/usa/taikun/rl-attack/rl_atk/attack-genai"
