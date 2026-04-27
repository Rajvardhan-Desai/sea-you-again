"""
app/worker/notify/sms.py
------------------------
Send SMS via Twilio REST API.
Raises on failure (caller should retry / capture).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_MAX_BODY = 320   # chars — fits 2 SMS segments with some room


def send_sms(
    to:       str,
    body:     str,
    settings,   # app.api.settings.Settings
) -> None:
    """
    Send an SMS message.
    `to` must be E.164 format (e.g. +919876543210).
    """
    if not settings.twilio_sid or not settings.twilio_token:
        log.warning("Twilio credentials not configured — SMS skipped.")
        return

    from twilio.rest import Client  # type: ignore[import]

    body = body[:_MAX_BODY]
    client = Client(settings.twilio_sid, settings.twilio_token)
    message = client.messages.create(
        body = body,
        from_ = settings.twilio_from,
        to    = to,
    )
    log.info(f"SMS sent → {to}  sid={message.sid}")
