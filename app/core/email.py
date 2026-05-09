import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Shared layout helpers
# ──────────────────────────────────────────────

def _base_template(content: str) -> str:
    """Wrap content inside the common email shell."""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MedIQ</title>
</head>
<body style="margin:0;padding:0;background-color:#0f1117;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0f1117;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

          <!-- HEADER -->
          <tr>
            <td style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);border-radius:16px 16px 0 0;padding:32px 40px;text-align:center;">
              <table cellpadding="0" cellspacing="0" style="margin:0 auto;">
                <tr>
                  <td style="background:rgba(255,255,255,0.15);border-radius:12px;padding:10px 14px;display:inline-block;">
                    <span style="font-size:22px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">
                      &#9889; MedIQ
                    </span>
                  </td>
                </tr>
              </table>
              <p style="margin:12px 0 0;color:rgba(255,255,255,0.75);font-size:13px;letter-spacing:0.5px;text-transform:uppercase;">
                Plateforme IA Médicale
              </p>
            </td>
          </tr>

          <!-- BODY -->
          <tr>
            <td style="background:#1a1d27;padding:40px;border-left:1px solid rgba(255,255,255,0.06);border-right:1px solid rgba(255,255,255,0.06);">
              {content}
            </td>
          </tr>

          <!-- FOOTER -->
          <tr>
            <td style="background:#13151f;border-radius:0 0 16px 16px;padding:24px 40px;border:1px solid rgba(255,255,255,0.06);border-top:none;text-align:center;">
              <p style="margin:0 0 6px;color:rgba(255,255,255,0.35);font-size:12px;">
                Cet e-mail a été envoyé automatiquement — merci de ne pas y répondre.
              </p>
              <p style="margin:0;color:rgba(255,255,255,0.25);font-size:11px;">
                © 2025 MedIQ · Tous droits réservés
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


# ──────────────────────────────────────────────
# Approval template
# ──────────────────────────────────────────────

def _approval_html(full_name: str) -> str:
    login_url = f"{settings.FRONTEND_URL}/login"
    content = f"""
      <!-- Icon -->
      <div style="text-align:center;margin-bottom:28px;">
        <div style="display:inline-block;background:linear-gradient(135deg,#10b981,#059669);border-radius:50%;width:64px;height:64px;line-height:64px;font-size:30px;text-align:center;">
          ✓
        </div>
      </div>

      <!-- Title -->
      <h1 style="margin:0 0 8px;color:#ffffff;font-size:24px;font-weight:700;text-align:center;letter-spacing:-0.5px;">
        Compte approuvé !
      </h1>
      <p style="margin:0 0 28px;color:rgba(255,255,255,0.5);font-size:14px;text-align:center;">
        Votre demande d'accès a été validée
      </p>

      <!-- Divider -->
      <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.5),transparent);margin-bottom:28px;"></div>

      <!-- Body text -->
      <p style="margin:0 0 16px;color:rgba(255,255,255,0.85);font-size:15px;line-height:1.7;">
        Bonjour <strong style="color:#ffffff;">{full_name}</strong>,
      </p>
      <p style="margin:0 0 16px;color:rgba(255,255,255,0.7);font-size:15px;line-height:1.7;">
        Nous avons le plaisir de vous informer que votre demande d'inscription sur
        <strong style="color:#8b5cf6;">MedIQ</strong> a été <strong style="color:#10b981;">approuvée</strong>
        par notre équipe d'administration.
      </p>
      <p style="margin:0 0 32px;color:rgba(255,255,255,0.7);font-size:15px;line-height:1.7;">
        Vous pouvez dès à présent accéder à la plateforme et commencer à utiliser
        l'ensemble des outils d'intelligence artificielle médicale mis à votre disposition.
      </p>

      <!-- Feature pills -->
      <table cellpadding="0" cellspacing="0" width="100%" style="margin-bottom:32px;">
        <tr>
          <td style="padding:4px;">
            <div style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:14px 16px;display:flex;align-items:center;">
              <span style="color:#8b5cf6;font-size:18px;margin-right:10px;">🧠</span>
              <span style="color:rgba(255,255,255,0.8);font-size:13px;">Entraînement de modèles IA sur vos datasets médicaux</span>
            </div>
          </td>
        </tr>
        <tr>
          <td style="padding:4px;">
            <div style="background:rgba(6,182,212,0.10);border:1px solid rgba(6,182,212,0.22);border-radius:10px;padding:14px 16px;">
              <span style="color:#06b6d4;font-size:18px;margin-right:10px;">📊</span>
              <span style="color:rgba(255,255,255,0.8);font-size:13px;">Analyse avancée et visualisation de vos données</span>
            </div>
          </td>
        </tr>
        <tr>
          <td style="padding:4px;">
            <div style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.22);border-radius:10px;padding:14px 16px;">
              <span style="color:#10b981;font-size:18px;margin-right:10px;">🎯</span>
              <span style="color:rgba(255,255,255,0.8);font-size:13px;">Prédictions interprétables et exportables</span>
            </div>
          </td>
        </tr>
      </table>

      <!-- CTA Button -->
      <div style="text-align:center;">
        <a href="{login_url}"
           style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#ffffff;font-size:15px;font-weight:600;text-decoration:none;padding:14px 36px;border-radius:10px;letter-spacing:0.2px;">
          Accéder à la plateforme →
        </a>
      </div>

      <p style="margin:28px 0 0;color:rgba(255,255,255,0.4);font-size:12px;text-align:center;">
        Bienvenue dans la communauté MedIQ 🎉
      </p>
    """
    return _base_template(content)


# ──────────────────────────────────────────────
# Rejection template
# ──────────────────────────────────────────────

def _rejection_html(full_name: str, reason: str | None) -> str:
    reason_block = ""
    if reason and reason.strip():
        reason_block = f"""
      <!-- Reason box -->
      <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-left:3px solid #ef4444;border-radius:10px;padding:16px 20px;margin-bottom:28px;">
        <p style="margin:0 0 4px;color:rgba(255,255,255,0.5);font-size:11px;text-transform:uppercase;letter-spacing:0.6px;">
          Motif communiqué
        </p>
        <p style="margin:0;color:rgba(255,255,255,0.85);font-size:14px;line-height:1.6;">
          {reason.strip()}
        </p>
      </div>
        """

    content = f"""
      <!-- Icon -->
      <div style="text-align:center;margin-bottom:28px;">
        <div style="display:inline-block;background:rgba(239,68,68,0.15);border:2px solid rgba(239,68,68,0.35);border-radius:50%;width:64px;height:64px;line-height:60px;font-size:28px;text-align:center;">
          ✕
        </div>
      </div>

      <!-- Title -->
      <h1 style="margin:0 0 8px;color:#ffffff;font-size:24px;font-weight:700;text-align:center;letter-spacing:-0.5px;">
        Demande non retenue
      </h1>
      <p style="margin:0 0 28px;color:rgba(255,255,255,0.5);font-size:14px;text-align:center;">
        Suite à l'examen de votre dossier
      </p>

      <!-- Divider -->
      <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(239,68,68,0.4),transparent);margin-bottom:28px;"></div>

      <!-- Body text -->
      <p style="margin:0 0 16px;color:rgba(255,255,255,0.85);font-size:15px;line-height:1.7;">
        Bonjour <strong style="color:#ffffff;">{full_name}</strong>,
      </p>
      <p style="margin:0 0 16px;color:rgba(255,255,255,0.7);font-size:15px;line-height:1.7;">
        Après examen de votre demande d'accès à la plateforme
        <strong style="color:#8b5cf6;">MedIQ</strong>, nous sommes dans
        l'obligation de vous informer que celle-ci n'a pas pu être validée à ce stade.
      </p>
      <p style="margin:0 0 28px;color:rgba(255,255,255,0.7);font-size:15px;line-height:1.7;">
        Cette décision peut être liée à des critères d'éligibilité ou à des informations
        manquantes dans votre dossier.
      </p>

      {reason_block}

      <!-- Info box -->
      <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:16px 20px;margin-bottom:28px;">
        <p style="margin:0;color:rgba(255,255,255,0.65);font-size:13px;line-height:1.7;">
          💡 Si vous pensez que cette décision est une erreur, ou si vous souhaitez
          fournir des informations complémentaires, veuillez contacter l'équipe
          d'administration directement.
        </p>
      </div>

      <p style="margin:0;color:rgba(255,255,255,0.45);font-size:13px;text-align:center;line-height:1.7;">
        Nous vous remercions de l'intérêt que vous portez à MedIQ<br/>
        et vous souhaitons bonne continuation.
      </p>
    """
    return _base_template(content)


# ──────────────────────────────────────────────
# Core send function
# ──────────────────────────────────────────────

def _send_email(to_email: str, subject: str, html_body: str) -> None:
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning("SMTP not configured — email not sent to %s", to_email)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
    msg["To"] = to_email

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, to_email, msg.as_string())
        logger.info("Email sent to %s | subject: %s", to_email, subject)
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def send_approval_email(to_email: str, full_name: str) -> None:
    subject = "✅ Votre compte MedIQ a été approuvé"
    html = _approval_html(full_name)
    _send_email(to_email, subject, html)


def send_rejection_email(to_email: str, full_name: str, reason: str | None = None) -> None:
    subject = "ℹ️ Décision concernant votre demande MedIQ"
    html = _rejection_html(full_name, reason)
    _send_email(to_email, subject, html)
