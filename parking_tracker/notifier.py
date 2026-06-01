"""Send iMessages from the local Mac via the Messages app (AppleScript)."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

_APPLESCRIPT = '''
on run {targetRecipient, messageText}
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy targetRecipient of targetService
        send messageText to targetBuddy
    end tell
end run
'''


def send_imessage(recipient: str, text: str) -> bool:
    """Send ``text`` to ``recipient`` (phone number or Apple ID email).

    Passing the recipient and body as ``osascript`` arguments avoids any string
    escaping or AppleScript-injection problems. Returns True on success.
    """
    proc = subprocess.run(
        ["osascript", "-e", _APPLESCRIPT, recipient, text],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error("Failed to send iMessage: %s", proc.stderr.strip())
        return False
    log.debug("Sent iMessage to %s", recipient)
    return True
