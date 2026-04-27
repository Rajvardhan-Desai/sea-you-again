"""
app/worker/notify/email.py
--------------------------
Send transactional email via SMTP (SendGrid relay or any SMTP server).
Uses stdlib smtplib — no extra dependency.
Retried by the caller (alert_engine wraps in tenacity).
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

log = logging.getLogger(__name__)


def send_email(
    to:       str,
    subject:  str,
    body:     str,
    settings,               # app.api.settings.Settings
    html_body: str = "",
) -> None:
    """
    Send a plain-text (+ optional HTML) email.
    Raises smtplib.SMTPException on failure (caller should retry / capture).
    """
    msg              = EmailMessage()
    msg["From"]      = settings.smtp_from
    msg["To"]        = to
    msg["Subject"]   = subject
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    log.info(f"Sending email → {to}: {subject}")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as s:
        s.ehlo()
        s.starttls()
        if settings.smtp_user and settings.smtp_pass:
            s.login(settings.smtp_user, settings.smtp_pass)
        s.send_message(msg)
    log.info(f"Email delivered → {to}")
