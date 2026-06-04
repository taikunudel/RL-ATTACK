#!/usr/bin/env python3
# PreToolUse(Bash) hook: FORBID rm / rmdir entirely (user rule, 2026-05-31).
# Reads the tool payload on stdin; denies any command that invokes rm or rmdir.
# Never delete — move to a delete/ folder and let the user decide.
import sys, json, re

try:
    data = json.load(sys.stdin)
except Exception:
    print("{}")
    sys.exit(0)

cmd = (data.get("tool_input", {}) or {}).get("command", "") or ""

# rm/rmdir as a command word: at start, after a shell separator (; & | ( ) { }),
# after whitespace, or after a path slash (/bin/rm). Avoids matching "warm",
# "--rmdir-flag", "chrm", or rm inside a quoted string argument that isn't a command.
if re.search(r'(^|[;&|(){}\s]|/)(rm|rmdir)(\s|$)', cmd):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "rm/rmdir is FORBIDDEN in this project (user rule). Do NOT delete "
                "anything. Instead: `mkdir -p delete && mv <thing> delete/` then let "
                "the USER decide whether to really delete it."
            ),
        }
    }))
else:
    print("{}")
