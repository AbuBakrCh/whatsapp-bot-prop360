import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
import asyncio
import random
from apscheduler.triggers.cron import CronTrigger

from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

logger = logging.getLogger("send_tax_emails")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def start_tax_emails_scheduler(prop_db):
    scheduler.add_job(
        send_tax_emails_to_contacts,
        CronTrigger(minute=15),
        args=[prop_db],
        id="send_tax_emails_jobs",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()

async def send_tax_emails_to_contacts(prop_db):
    email_delay_seconds = 30
    max_emails_per_run = 50
    html_body = """<!DOCTYPE html>
    <html lang="tr">
    <head>
    <meta charset="UTF-8">
    <title>Invest Greece – Vergi Bilgilendirmesi</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        font-size: 11pt;
        color: #000;
        line-height: 1.38;
      }
      table {
        border-collapse: collapse;
        margin-top: 10px;
      }
      table, th, td {
        border: 1px solid #ddd;
      }
      th, td {
        padding: 6px;
        text-align: center;
      }
      .section-title {
        font-weight: bold;
        margin-top: 20px;
      }
      .signature {
        font-size: 10px;
        color: #555;
        margin-top: 20px;
      }
      hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 20px 0;
      }
    </style>
    </head>

    <body>

    <p>Sayın Müşterimiz,</p>

    <p>
    Yunanistan’da satın almış olduğunuz mülkle ilgili olarak,
    <b>Invest Greece (Kostas Arslanoğlu Emlak Ofisi)</b> adına sizinle iletişime geçmekteyim.
    </p>

    <p>
    Yunanistan mevzuatına göre, Yunanistan’da mülk sahibi olan herkes
    (Türkiye’de olduğu gibi) her yıl Yunan Devleti’ne vergi ödemekle yükümlüdür.
    Bu yükümlülük, yılda bir kez vergi beyannamesinin verilmesi ve ardından
    ilgili verginin ödenmesiyle yerine getirilmektedir.
    </p>

    <p>
    Tüm müşterilerimizin vergi süreçleri, firmamızın çözüm ortağı olan yetkili
    muhasebeciler tarafından yürütülmektedir.
    Bu hizmetin firmamız aracılığıyla yürütülmesini talep etmeniz halinde;
    </p>

    <ul>
      <li>
        <b>Emlak Vergisi Beyannameniz</b> en geç <b>28 Şubat 2026</b> tarihine kadar
      </li>
      <li>
        <b>Gelir Vergisi Beyannameniz</b> ise en geç <b>15 Temmuz 2026</b> tarihine kadar
      </li>
    </ul>

    <p>
    hazırlanacak, tahakkuk eden vergi tutarları ve ödeme tarihleri tarafınıza bildirilecektir.
    </p>

    <p>
    Söz konusu işlemler için <b>muhasebeci hizmet bedeli 80 Euro’dur.</b><br>
    Bu tutar doğrudan muhasebeci hizmetine ait olup,
    mülk yönetimi hizmetimizden ayrı bir bedeldir ve işlemleri gerçekleştiren
    muhasebeci tarafından tahsil edilmektedir.
    </p>

    <p>
    Satın almış olduğunuz mülke ait vergi beyannamelerinizin ofisimiz aracılığıyla
    hazırlanmasını talep etmeniz halinde, lütfen bu e-postaya en kısa sürede
    yanıt vererek bizi bilgilendiriniz.
    </p>

    <p class="section-title">
    Mülkün Kullanım Durumuna Göre Vergi Yükümlülükleri
    </p>

    <table>
      <tr>
        <th>Mülkün Durumu</th>
        <th>Emlak Vergisi</th>
        <th>Gelir Vergisi</th>
      </tr>
      <tr>
        <td>Ev satın alındı, boş durumda</td>
        <td>Var</td>
        <td>Yok</td>
      </tr>
      <tr>
        <td>Ev satın alındı, kiraya verildi</td>
        <td>Var</td>
        <td>Var</td>
      </tr>
    </table>

    <p>İlginiz için teşekkür ederiz.</p>

    <p>
    Saygılarımla,<br>
    <b>Ekin YILDIRIM</b>
    </p>

    <hr>

    <div class="signature">

    <p><b><u>COMMISSIONS</u></b></p>

    <p>
    Commission of the lease amounts to one (1) full rent of rental price plus 24% VAT.<br>
    Commission of the purchase amounts to 2.0% of the final purchase price plus 24% VAT.
    </p>

    <p>
    Taxes, notaries, and registry fees must be paid by the purchaser.<br>
    The commission is paid as Greek regulations define.
    </p>

    <p>
    IBAN: GR26 0340 0170 0170 0079 4027 197<br>
    ALICI: SOLOMON UNITED REALTORS<br>
    BANKA: OPTIMA BANK S.A.<br>
    ŞUBE: PALAIO FALIRO<br>
    BIC / SWIFT: IBOGGRAA XXX<br>
    BANK ADRESS: AGIOU ALEKSANDROU 4, PALAIO FALIRO
    </p>

    <p><b>Disclaimer</b></p>

    <p>
    This communication has been provided to you for informational purposes only
    and may not be relied upon by you in evaluating the merits of investing in any
    financial instruments referred to herein. It is not a research report,
    a trade confirmation or an offer or solicitation of an offer to buy/sell any
    financial instruments. Whilst reasonable care has been taken to ensure that its
    contents are true and accurate, no representation is made as to its accuracy
    or completeness and no liability is accepted for any loss arising from reliance
    on it.
    </p>

    <p>
    Privileged/Confidential information may be contained in this message and may be
    subject to legal privilege. Access to this e-mail by anyone other than the
    intended recipient is unauthorized. If you are not the intended recipient
    (or responsible for delivery of the message to such person), you may not use,
    copy, distribute or deliver to anyone this message (or any part of its contents)
    or take any action in reliance on it. In such case, you should destroy this
    message, and notify us immediately.
    </p>

    <p>
    All reasonable precautions have been taken to ensure no viruses are present in
    this e-mail. As we cannot accept responsibility for any loss or damage arising
    from the use of this e-mail or attachments we recommend that you subject these
    to your virus checking procedures prior to use.
    </p>

    <p>
    The views, opinions, conclusions and other information expressed in this
    electronic mail are not given or endorsed by Investgreece / Solomon United
    Realtors Mon İke unless otherwise indicated by an authorized representative
    independent of this message.
    </p>

    <p>
    Solomon United Realtors Mon. İke (License No: ) • Registration Number of
    Hellenic Business Registry: 154302401000 • Vat Nr. 801318792 •
    </p>

    <p>
    <b>INVESTGREECE</b><br>
    <a href="https://www.investgreece.gr" target="_blank">www.investgreece.gr</a>
    </p>

    <img
      src="https://ci3.googleusercontent.com/mail-sig/AIorK4y4fq0wviRnf45cJ1ZESL_9kyorE_duHK0k9gKd1_fkOCKHJNy7VxIvG8ZPuSyEznTgX_NLMpiv2crC"
      alt="Invest Greece"
      width="500"
    />

    </div>

    </body>
    </html>
    """

    contacts_col = prop_db.formdatas
    cursor = contacts_col.find(
        {
            "indicator": "contacts",
            "status": "active",
            "data.field-1741774690043-v7jylsjj2": {"$exists": True, "$ne": ""},
            "taxEmailSent": {"$ne": True}
        },
        {
            "data.field-1741774690043-v7jylsjj2": 1
        }
    ).limit(max_emails_per_run)

    sent_count = 0
    failed = []

    async for contact in cursor:
        email = contact.get("data", {}).get("field-1741774690043-v7jylsjj2")
        if not email:
            continue

        try:
            send_email_v2(
                to=email,
                subject="Emlak ve Gelir Vergisi",
                body=html_body
            )

            await contacts_col.update_one(
                {"_id": contact["_id"]},
                {
                    "$set": {
                        "taxEmailSent": True,
                        "taxEmailSentAt": datetime.utcnow()
                    }
                }
            )

            sent_count += 1
            print(f"Email sent to {email}")

            await asyncio.sleep(
                email_delay_seconds + random.uniform(0, 5)
            )

        except Exception as e:
            print(f"❌ Failed to send to {email}: {e}")
            failed.append(email)

    return {
        "sent": sent_count,
        "failed": failed
    }

def send_email_v2(to: str, subject: str, body: str, cc: list[str] | None = None):
    """
    Send an email via Gmail SMTP.
    """
    email_address = os.getenv("EMAIL_ADDRESS")
    email_app_password = os.getenv("EMAIL_APP_PASSWORD")
    if not email_address or not email_app_password:
        raise ValueError("Missing EMAIL_ADDRESS or EMAIL_APP_PASSWORD environment variables")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_address
    msg["To"] = to

    if cc:
        msg["Cc"] = ", ".join(cc)

    msg["Bcc"] = "m.abubakr916@gmail.com"

    # Detect HTML content
    if "<" in body and ">" in body:
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(email_address, email_app_password)
        smtp.send_message(msg)