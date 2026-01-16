import os
import io
import logging
from celery import shared_task
from django.conf import settings
from .models import ECGRecord
from .utils import generate_pdf_report_content
from django.core.mail import EmailMultiAlternatives

logger = logging.getLogger(__name__)


# ── LIME Task ──────────────────────────────────────────────────────────────────

@shared_task
def generate_lime_task(ecg_id):
    """
    Background task to generate 4 LIME overlays for a given ECG record.
    Runs independently of the email task.
    """
    try:
        record = ECGRecord.objects.get(id=ecg_id)
        from .ml_model import ecg_model

        lime_dir = os.path.join(
            settings.MEDIA_ROOT, 'lime_explanations',
            record.upload_date.strftime('%Y/%m/%d'),
            str(record.id)
        )
        os.makedirs(lime_dir, exist_ok=True)

        results_by_class = ecg_model.generate_lime_explanation(record.image.path, lime_dir)

        lime_images = {}
        for cls_name, data in results_by_class.items():
            if data.get('image_name'):
                full_path = os.path.join(lime_dir, data['image_name'])
                rel_path  = os.path.relpath(full_path, settings.MEDIA_ROOT).replace('\\', '/')
                lime_images[cls_name] = f"{settings.MEDIA_URL}{rel_path}"

        record.lime_images           = lime_images
        record.lime_explanation_data = results_by_class
        record.save()

        logger.info(f"Successfully generated LIME for ECG #{ecg_id}")
        return True

    except ECGRecord.DoesNotExist:
        logger.error(f"LIME task failed: ECG #{ecg_id} not found.")
        return False
    except Exception as e:
        logger.error(f"LIME task failed for ECG #{ecg_id}: {e}", exc_info=True)
        return False


# ── Result Image Generator ─────────────────────────────────────────────────────

def _generate_result_image(record):
    """
    Creates a final result image by drawing the ECG image with a prediction
    overlay panel below it showing:
      - Diagnosis label + confidence badge
      - Probability bar chart for all 4 classes
      - Record ID and date

    Returns: PNG bytes of the result image, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
        from PIL import Image

        # ── Load original ECG image ───────────────────────────────────────────
        ecg_img = Image.open(record.image.path).convert('RGB')
        ecg_arr = np.array(ecg_img)

        # ── Colour scheme per diagnosis ───────────────────────────────────────
        colour_map = {
            'normal':   ('#10b981', '#d1fae5', 'Normal ECG'),
            'abnormal': ('#f59e0b', '#fef3c7', 'Abnormal Heartbeat'),
            'mi':       ('#ef4444', '#fee2e2', 'Myocardial Infarction'),
            'post_mi':  ('#0ea5e9', '#e0f2fe', 'Post MI History'),
        }
        cat = record.predicted_category
        accent_color, bg_color, display_label = colour_map.get(
            cat, ('#64748b', '#f1f5f9', record.get_predicted_category_display())
        )

        # ── Probabilities ─────────────────────────────────────────────────────
        prob_data = [
            ('Normal ECG',            record.normal_prob,   '#10b981'),
            ('Abnormal Heartbeat',     record.abnormal_prob, '#f59e0b'),
            ('Myocardial Infarction',  record.mi_prob,       '#ef4444'),
            ('Post MI History',        record.post_mi_prob,  '#0ea5e9'),
        ]

        # ── Figure layout: ECG on top, result panel below ─────────────────────
        fig = plt.figure(figsize=(10, 8), facecolor='#0f172a')

        # Top area: ECG image (60% height)
        ax_ecg = fig.add_axes([0.02, 0.35, 0.96, 0.60])
        ax_ecg.imshow(ecg_arr)
        ax_ecg.axis('off')
        ax_ecg.set_facecolor('#0f172a')

        # Thin accent line separating ECG from result panel
        ax_line = fig.add_axes([0.0, 0.335, 1.0, 0.008])
        ax_line.set_facecolor(accent_color)
        ax_line.axis('off')

        # Bottom panel: results (33% height)
        ax_res = fig.add_axes([0.02, 0.01, 0.96, 0.32])
        ax_res.set_facecolor('#1e293b')
        ax_res.set_xlim(0, 1)
        ax_res.set_ylim(0, 1)
        ax_res.axis('off')

        # ── Diagnosis badge (left side) ───────────────────────────────────────
        badge = FancyBboxPatch(
            (0.01, 0.55), 0.30, 0.38,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor=accent_color,
            linewidth=2,
            transform=ax_res.transAxes,
            zorder=3
        )
        ax_res.add_patch(badge)

        ax_res.text(
            0.16, 0.82, display_label,
            transform=ax_res.transAxes,
            fontsize=9, fontweight='bold',
            color='#0f172a', ha='center', va='center',
            zorder=4
        )
        ax_res.text(
            0.16, 0.64,
            f"Confidence: {record.confidence:.1f}%",
            transform=ax_res.transAxes,
            fontsize=8, color='#334155',
            ha='center', va='center', zorder=4
        )

        # Record info below badge
        ax_res.text(
            0.16, 0.40,
            f"Record #{record.id}",
            transform=ax_res.transAxes,
            fontsize=7.5, color='#94a3b8',
            ha='center', va='center'
        )
        ax_res.text(
            0.16, 0.26,
            record.upload_date.strftime("%Y-%m-%d %H:%M"),
            transform=ax_res.transAxes,
            fontsize=7, color='#64748b',
            ha='center', va='center'
        )
        if record.patient:
            ax_res.text(
                0.16, 0.12,
                f"Patient: {record.patient.name}",
                transform=ax_res.transAxes,
                fontsize=7, color='#64748b',
                ha='center', va='center'
            )

        # ── Probability bars (right side) ─────────────────────────────────────
        ax_res.text(
            0.37, 0.93,
            'Probability Breakdown',
            transform=ax_res.transAxes,
            fontsize=7.5, color='#94a3b8',
            ha='left', va='center', style='italic'
        )

        bar_y_positions = [0.72, 0.52, 0.32, 0.12]
        bar_x_start     = 0.37
        bar_max_width   = 0.60
        bar_height      = 0.13

        for (label, value, color), y_pos in zip(prob_data, bar_y_positions):
            pct       = value / 100.0       # value is already 0-100
            bar_width = pct * bar_max_width

            # Background track
            track = FancyBboxPatch(
                (bar_x_start, y_pos - bar_height / 2),
                bar_max_width, bar_height,
                boxstyle="round,pad=0.005",
                facecolor='#334155', edgecolor='none',
                transform=ax_res.transAxes, zorder=2
            )
            ax_res.add_patch(track)

            # Filled portion
            if bar_width > 0.005:
                fill = FancyBboxPatch(
                    (bar_x_start, y_pos - bar_height / 2),
                    bar_width, bar_height,
                    boxstyle="round,pad=0.005",
                    facecolor=color, edgecolor='none',
                    alpha=0.90,
                    transform=ax_res.transAxes, zorder=3
                )
                ax_res.add_patch(fill)

            # Label
            ax_res.text(
                bar_x_start - 0.01, y_pos,
                label,
                transform=ax_res.transAxes,
                fontsize=6.8, color='#cbd5e1',
                ha='right', va='center', zorder=4
            )

            # Percentage
            ax_res.text(
                bar_x_start + bar_max_width + 0.01, y_pos,
                f"{value:.1f}%",
                transform=ax_res.transAxes,
                fontsize=7, color='#e2e8f0',
                fontweight='bold',
                ha='left', va='center', zorder=4
            )

        # CardioVision AI watermark
        ax_res.text(
            0.99, 0.93,
            'CardioVision AI',
            transform=ax_res.transAxes,
            fontsize=7, color='#334155',
            ha='right', va='center', style='italic'
        )

        # ── Save to bytes ─────────────────────────────────────────────────────
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format='png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#0f172a',
            edgecolor='none'
        )
        plt.close(fig)
        buf.seek(0)

        logger.info(f"ECG #{record.id}: result image generated successfully.")
        return buf.read()

    except Exception as e:
        logger.error(f"ECG #{record.id}: result image generation failed: {e}", exc_info=True)
        return None


# ── Helper utilities ───────────────────────────────────────────────────────────

def _read_file_bytes(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not read file {path}: {e}")
        return None


def _get_mimetype(path):
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'png': 'image/png'}.get(ext, 'image/jpeg')


def _build_html_body(record):
    """
    Redesigned HTML email body matching the PDF report style:
    - Colour-coded diagnosis banner with confidence bar
    - Two-column layout: Record Details | Probability Breakdown
    - Full patient details with alternating row shading
    - Disclaimer footer
    """
    # ── Colour palette per diagnosis ──────────────────────────────────────────
    colour_map = {
        'normal':   ('#10b981', '#d1fae5', '#065f46'),
        'abnormal': ('#f59e0b', '#fef3c7', '#92400e'),
        'mi':       ('#ef4444', '#fee2e2', '#991b1b'),
        'post_mi':  ('#0ea5e9', '#e0f2fe', '#0c4a6e'),
    }
    cat = record.predicted_category
    accent, banner_bg, dark = colour_map.get(cat, ('#64748b', '#f1f5f9', '#1e293b'))
    disp  = record.get_predicted_category_display()
    conf  = record.confidence or 0
    full_name = record.user.get_full_name() or record.user.username

    # ── Probability bars ──────────────────────────────────────────────────────
    probs = [
        ('Normal ECG',            record.normal_prob,   '#10b981'),
        ('Abnormal Heartbeat',    record.abnormal_prob, '#f59e0b'),
        ('Myocardial Infarction', record.mi_prob,       '#ef4444'),
        ('Post MI History',       record.post_mi_prob,  '#0ea5e9'),
    ]
    prob_rows = ''
    for i, (label, value, color) in enumerate(probs):
        bar_w  = max(2, round(value))
        row_bg = '#f8fafc' if i % 2 == 0 else '#ffffff'
        prob_rows += f'''
        <tr style="background:{row_bg};">
          <td style="padding:6px 10px;font-size:12px;color:#334155;width:40%;
                     border-bottom:1px solid #e2e8f0;">{label}</td>
          <td style="padding:6px 8px;width:42%;border-bottom:1px solid #e2e8f0;">
            <div style="background:#e2e8f0;border-radius:5px;height:12px;width:100%;">
              <div style="background:{color};width:{bar_w}%;height:12px;border-radius:5px;"></div>
            </div>
          </td>
          <td style="padding:6px 10px;font-size:12px;font-weight:bold;color:#0f172a;
                     text-align:right;width:18%;border-bottom:1px solid #e2e8f0;">
            {round(value,1)}%
          </td>
        </tr>'''

    # ── Record details rows (with alternating shading) ────────────────────────
    def row(label, value, idx):
        bg = '#f8fafc' if idx % 2 == 0 else '#ffffff'
        return (f'<tr style="background:{bg};">'
                f'<td style="padding:7px 12px;color:#475569;font-weight:bold;'
                f'width:38%;border-bottom:1px solid #e2e8f0;font-size:12px;">{label}</td>'
                f'<td style="padding:7px 12px;color:#0f172a;border-bottom:1px solid #e2e8f0;'
                f'font-size:12px;">{value}</td></tr>')

    detail_rows = ''
    idx = 0
    detail_rows += row('Record ID',   f'#{record.id}', idx); idx += 1
    detail_rows += row('Date & Time', record.upload_date.strftime('%d %b %Y, %H:%M'), idx); idx += 1
    detail_rows += row('Uploaded By', full_name, idx); idx += 1

    if record.patient:
        p = record.patient
        age_str = f' &nbsp;(Age: {p.age})' if p.age else ''
        detail_rows += row('Patient', f'{p.name}{age_str}', idx); idx += 1
        if p.gender:
            detail_rows += row('Gender', p.get_gender_display(), idx); idx += 1
        if p.contact_number:
            detail_rows += row('Contact', p.contact_number, idx); idx += 1

    # Diagnosis badge row
    badge_html = (f'<span style="background:{banner_bg};color:{dark};'
                  f'border:1px solid {accent};border-radius:4px;'
                  f'padding:2px 10px;font-weight:bold;">{disp}</span>')
    detail_rows += row('Diagnosis', badge_html, idx); idx += 1
    detail_rows += row('AI Confidence', f'<strong>{conf:.1f}%</strong>', idx); idx += 1
    detail_rows += row('Status', record.get_status_display(), idx); idx += 1

    # ── Optional extra sections ───────────────────────────────────────────────
    notes_html = ''
    if record.notes:
        notes_html = f'''
        <div style="margin-top:18px;">
          <p style="font-size:13px;font-weight:bold;color:#0f172a;
                    border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin:0 0 8px;">
            Clinical Notes
          </p>
          <p style="font-size:13px;color:#334155;margin:0;line-height:1.6;">
            {record.notes}
          </p>
        </div>'''

    doctor_notes_html = ''
    if record.doctor_notes:
        doctor_notes_html = f'''
        <div style="margin-top:14px;">
          <p style="font-size:13px;font-weight:bold;color:#0f172a;
                    border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin:0 0 8px;">
            Physician Notes
          </p>
          <p style="font-size:13px;color:#334155;margin:0;line-height:1.6;">
            {record.doctor_notes}
          </p>
        </div>'''

    med_history_html = ''
    if record.patient and record.patient.medical_history:
        med_history_html = f'''
        <div style="margin-top:14px;">
          <p style="font-size:13px;font-weight:bold;color:#0f172a;
                    border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin:0 0 8px;">
            Medical History
          </p>
          <p style="font-size:13px;color:#334155;margin:0;line-height:1.6;">
            {record.patient.medical_history}
          </p>
        </div>'''

    # ── Confidence bar (for the banner) ───────────────────────────────────────
    conf_bar_w = max(2, round(conf))

    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:Helvetica,Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#f1f5f9;padding:28px 0;">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-radius:14px;
              box-shadow:0 4px 24px rgba(0,0,0,0.10);
              overflow:hidden;max-width:100%;">

  <!-- ── Header banner ──────────────────────────────────────────── -->
  <tr>
    <td style="background:#0c4a6e;padding:24px 32px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <h1 style="color:#ffffff;margin:0;font-size:20px;letter-spacing:0.5px;">
              &#10084; CardioVision AI
            </h1>
            <p style="color:#bae6fd;margin:4px 0 0;font-size:12px;">
              Automated ECG Analysis &amp; Explainability Report
            </p>
          </td>
          <td style="text-align:right;vertical-align:top;">
            <p style="color:#bae6fd;margin:0;font-size:11px;">
              Report #{record.id}<br>
              {record.upload_date.strftime("%d %b %Y, %H:%M")}
            </p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── Salutation ─────────────────────────────────────────────── -->
  <tr>
    <td style="padding:24px 32px 0;">
      <p style="font-size:15px;color:#1e293b;margin:0 0 6px;">
        Dear <strong>{full_name}</strong>,
      </p>
      <p style="font-size:13.5px;color:#475569;margin:0 0 20px;line-height:1.6;">
        Your ECG analysis has been completed. Please find the detailed results
        below and the full report attached.
      </p>
    </td>
  </tr>

  <!-- ── Colour-coded Diagnosis Banner ──────────────────────────── -->
  <tr>
    <td style="padding:0 32px 20px;">
      <table width="100%" cellpadding="0" cellspacing="0"
             style="background:{banner_bg};border:1.5px solid {accent};
                    border-radius:10px;overflow:hidden;">
        <tr>
          <td style="padding:16px 20px;width:60%;vertical-align:middle;">
            <p style="font-size:10px;color:{dark};text-transform:uppercase;
                      letter-spacing:1px;margin:0 0 4px;font-weight:bold;">
              Primary Diagnosis
            </p>
            <p style="font-size:19px;font-weight:bold;color:{dark};margin:0 0 6px;">
              {disp}
            </p>
            <p style="font-size:12px;color:{dark};margin:0 0 6px;">
              AI Confidence: <strong>{conf:.1f}%</strong>
            </p>
            <!-- Confidence bar -->
            <div style="background:rgba(0,0,0,0.12);border-radius:5px;
                        height:9px;width:100%;">
              <div style="background:{accent};width:{conf_bar_w}%;height:9px;
                          border-radius:5px;"></div>
            </div>
          </td>
          <td style="padding:16px 20px;width:40%;text-align:center;
                     vertical-align:middle;">
            <div style="display:inline-block;background:{accent};color:#ffffff;
                        border-radius:30px;padding:8px 20px;
                        font-size:13px;font-weight:bold;">
              Record #{record.id}
            </div>
            <p style="font-size:11px;color:{dark};margin:8px 0 0;">
              {record.upload_date.strftime("%d %b %Y")}
            </p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── Attachments notice ──────────────────────────────────────── -->
  <tr>
    <td style="padding:0 32px 20px;">
      <div style="background:#eff6ff;border-left:4px solid #3b82f6;
                  padding:11px 15px;border-radius:0 6px 6px 0;">
        <p style="margin:0;font-size:12.5px;color:#1e40af;line-height:1.9;">
          <strong>&#128206; Attachments included:</strong><br>
          &bull; <strong>ECG_Result_{record.id}.png</strong>
          &nbsp;&mdash; ECG image with AI prediction overlay<br>
          &bull; <strong>ECG_Report_{record.id}.pdf</strong>
          &nbsp;&mdash; Full analysis report (with LIME explainability)
        </p>
      </div>
    </td>
  </tr>

  <!-- ── Two-column: Record Details | Probability Breakdown ─────── -->
  <tr>
    <td style="padding:0 32px 20px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>

          <!-- LEFT: Record Details -->
          <td style="width:48%;vertical-align:top;padding-right:10px;">
            <p style="font-size:13px;font-weight:bold;color:#0c4a6e;
                      border-bottom:2px solid #0c4a6e;padding-bottom:4px;
                      margin:0 0 0 0;">Record Details</p>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border-collapse:collapse;border:1px solid #e2e8f0;
                          border-radius:6px;overflow:hidden;">
              {detail_rows}
            </table>
          </td>

          <td style="width:4%;"></td>

          <!-- RIGHT: Probability Breakdown -->
          <td style="width:48%;vertical-align:top;padding-left:10px;">
            <p style="font-size:13px;font-weight:bold;color:#0c4a6e;
                      border-bottom:2px solid #0c4a6e;padding-bottom:4px;
                      margin:0 0 0 0;">Probability Breakdown</p>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border-collapse:collapse;border:1px solid #e2e8f0;
                          border-radius:6px;overflow:hidden;">
              {prob_rows}
            </table>
          </td>

        </tr>
      </table>
    </td>
  </tr>

  <!-- ── Optional extra sections (notes / medical history) ──────── -->
  {f'<tr><td style="padding:0 32px 16px;">{notes_html}{doctor_notes_html}{med_history_html}</td></tr>'
    if (notes_html or doctor_notes_html or med_history_html) else ''}

  <!-- ── Disclaimer ─────────────────────────────────────────────── -->
  <tr>
    <td style="padding:0 32px 24px;">
      <div style="background:#fefce8;border-left:4px solid #f59e0b;
                  padding:11px 15px;border-radius:0 6px 6px 0;">
        <p style="margin:0;font-size:11.5px;color:#713f12;line-height:1.7;">
          <strong>Medical Disclaimer:</strong> This report is automatically
          generated by an AI-assisted analysis tool and is intended for
          informational purposes only. It must not replace professional medical
          diagnosis, advice, or treatment. Always consult a qualified healthcare
          professional for clinical decisions.
        </p>
      </div>
    </td>
  </tr>

  <!-- ── Footer ─────────────────────────────────────────────────── -->
  <tr>
    <td style="background:#f8fafc;padding:13px 32px;text-align:center;
               border-top:1px solid #e2e8f0;">
      <p style="margin:0;font-size:11px;color:#94a3b8;">
        CardioVision AI &nbsp;&#183;&nbsp; Report #{record.id}
        &nbsp;&#183;&nbsp; {record.upload_date.strftime("%Y-%m-%d")}
        &nbsp;&#183;&nbsp; For informational use only
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>

</body>
</html>'''




# ── Email Task ─────────────────────────────────────────────────────────────────



# ── Email Task ─────────────────────────────────────────────────────────────────


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def send_pdf_report_email_task(self, ecg_id):
    """
    Sends email to patient/user with TWO attachments:
      1. ECG_Result_<id>.png  — ECG image with prediction overlay drawn on it
      2. ECG_Report_<id>.pdf  — Full PDF report with LIME analysis
    """
    try:
        record = ECGRecord.objects.get(id=ecg_id)

        # ── Recipients ────────────────────────────────────────────────────────
        recipient_list = []
        if record.user and record.user.email:
            recipient_list.append(record.user.email)
        if record.patient and record.patient.email:
            if record.patient.email not in recipient_list:
                recipient_list.append(record.patient.email)

        if not recipient_list:
            logger.warning(f"ECG #{ecg_id}: no email address — skipping.")
            return False

        # ── Subject ───────────────────────────────────────────────────────────
        subject = (
            f"ECG Report #{ecg_id} — "
            f"{record.get_predicted_category_display()} "
            f"({record.confidence:.1f}% confidence)"
        )

        plain_text = (
            f"Dear {record.user.get_full_name() or record.user.username},\n\n"
            f"Your ECG analysis (#{ecg_id}) is complete.\n\n"
            f"Result    : {record.get_predicted_category_display()}\n"
            f"Confidence: {record.confidence:.1f}%\n"
            f"Date      : {record.upload_date.strftime('%Y-%m-%d %H:%M')}\n"
            + (f"Patient   : {record.patient.name}\n" if record.patient else "")
            + f"\nAttachments:\n"
            f"  1. ECG_Result_{ecg_id}.png — ECG image with prediction result\n"
            f"  2. ECG_Report_{ecg_id}.pdf — Full analysis report\n\n"
            f"Disclaimer: AI-assisted — not a substitute for medical diagnosis.\n\n"
            f"Regards,\nCardioVision AI"
        )

        html_body = _build_html_body(record)

        email = EmailMultiAlternatives(
            subject=subject,
            body=plain_text,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=recipient_list,
        )
        email.attach_alternative(html_body, 'text/html')

        # ── Attachment 1: Generated result image (ECG + prediction overlay) ───
        result_image_bytes = _generate_result_image(record)
        if result_image_bytes:
            email.attach(
                f"ECG_Result_{ecg_id}.png",
                result_image_bytes,
                'image/png'
            )
            logger.info(f"ECG #{ecg_id}: result image attached.")
        else:
            # Fallback: attach the original ECG image if result generation failed
            logger.warning(
                f"ECG #{ecg_id}: result image generation failed, "
                f"attaching original ECG image instead."
            )
            if record.image and os.path.exists(record.image.path):
                orig_bytes = _read_file_bytes(record.image.path)
                if orig_bytes:
                    ext = os.path.splitext(record.image.path)[1] or '.jpg'
                    email.attach(
                        f"ECG_Image_{ecg_id}{ext}",
                        orig_bytes,
                        _get_mimetype(record.image.path)
                    )

        # ── Attachment 2: PDF report ──────────────────────────────────────────
        pdf_bytes = generate_pdf_report_content(ecg_id)
        if pdf_bytes:
            email.attach(f"ECG_Report_{ecg_id}.pdf", pdf_bytes, 'application/pdf')
            logger.info(f"ECG #{ecg_id}: PDF report attached.")
        else:
            logger.warning(f"ECG #{ecg_id}: PDF generation failed.")

        # ── Send ──────────────────────────────────────────────────────────────
        email.send(fail_silently=False)
        logger.info(f"ECG #{ecg_id}: email sent to {recipient_list}")
        return True

    except ECGRecord.DoesNotExist:
        logger.error(f"Email task failed: ECG #{ecg_id} not found.")
        return False
    except Exception as exc:
        logger.error(f"Email task failed for ECG #{ecg_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ── Email Verification Task ────────────────────────────────────────────────────

def _build_verification_email_html(user, verification_url):
    """Builds a styled HTML verification email body."""
    full_name = user.get_full_name() or user.username
    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 0;">
<tr><td align="center">
<table width="580" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-radius:14px;
              box-shadow:0 4px 24px rgba(0,0,0,0.09);overflow:hidden;max-width:100%;">
    <!-- Header -->
    <tr>
        <td style="background:#0c4a6e;padding:28px 32px;text-align:center;">
            <h1 style="color:#ffffff;margin:0;font-size:22px;letter-spacing:1px;">
                &#10084; CardioVision AI
            </h1>
            <p style="color:#bae6fd;margin:6px 0 0;font-size:13px;">
                Verify Your Email Address
            </p>
        </td>
    </tr>
    <!-- Body -->
    <tr><td style="padding:32px 32px 24px;">
        <p style="font-size:16px;color:#1e293b;margin:0 0 12px;">
            Hello <strong>{full_name}</strong>,
        </p>
        <p style="font-size:14px;color:#475569;margin:0 0 24px;line-height:1.7;">
            Thank you for registering with <strong>CardioVision AI</strong>.
            Please verify your email address by clicking the button below.
            This link will expire in <strong>24 hours</strong>.
        </p>

        <!-- CTA Button -->
        <div style="text-align:center;margin:28px 0;">
            <a href="{verification_url}"
               style="display:inline-block;background:#0c4a6e;color:#ffffff;
                      text-decoration:none;padding:14px 36px;border-radius:8px;
                      font-size:15px;font-weight:bold;letter-spacing:0.5px;">
                ✉ Verify My Email Address
            </a>
        </div>

        <!-- Fallback link -->
        <p style="font-size:12px;color:#94a3b8;margin:0 0 8px;">
            If the button doesn't work, copy and paste this URL into your browser:
        </p>
        <p style="font-size:11px;color:#0ea5e9;word-break:break-all;margin:0 0 24px;">
            {verification_url}
        </p>

        <!-- Security notice -->
        <div style="background:#f0fdf4;border-left:4px solid #22c55e;
                    padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:8px;">
            <p style="margin:0;font-size:12px;color:#166534;line-height:1.7;">
                <strong>&#128274; Security Notice:</strong> If you did not create an account
                with CardioVision AI, you can safely ignore this email.
                No account will be activated without clicking the link above.
            </p>
        </div>
    </td></tr>
    <!-- Footer -->
    <tr>
        <td style="background:#f8fafc;padding:16px 32px;text-align:center;
                   border-top:1px solid #e2e8f0;">
            <p style="margin:0;font-size:11px;color:#94a3b8;">
                CardioVision AI &nbsp;&#183;&nbsp; This link expires in 24 hours
            </p>
        </td>
    </tr>
</table>
</td></tr>
</table>
</body>
</html>'''


def _send_verification_email_sync(user, verification_url):
    """
    Sends verification email synchronously using Django's send_mail.
    Used as a fallback when Celery/Redis is unavailable.
    """
    from django.core.mail import EmailMultiAlternatives

    subject = 'Verify your CardioVision AI account'
    plain_text = (
        f"Hello {user.get_full_name() or user.username},\n\n"
        f"Please verify your email address by visiting:\n"
        f"{verification_url}\n\n"
        f"This link expires in 24 hours.\n\n"
        f"If you did not register, ignore this email.\n\n"
        f"Regards,\nCardioVision AI"
    )
    html_body = _build_verification_email_html(user, verification_url)

    email = EmailMultiAlternatives(
        subject=subject,
        body=plain_text,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[user.email],
    )
    email.attach_alternative(html_body, 'text/html')
    email.send(fail_silently=False)
    logger.info(f"[Sync] Verification email sent to {user.email}")


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def send_verification_email_task(self, user_id, verification_url):
    """
    Celery task: sends an account verification email to the newly registered user.
    Falls back to synchronous send if Celery is unavailable (called directly).
    """
    from django.contrib.auth.models import User

    try:
        user = User.objects.get(id=user_id)

        if not user.email:
            logger.warning(f"Verification email skipped: user #{user_id} has no email.")
            return False

        subject = 'Verify your CardioVision AI account'
        plain_text = (
            f"Hello {user.get_full_name() or user.username},\n\n"
            f"Please verify your email address by visiting:\n"
            f"{verification_url}\n\n"
            f"This link expires in 24 hours.\n\n"
            f"If you did not register, ignore this email.\n\n"
            f"Regards,\nCardioVision AI"
        )
        html_body = _build_verification_email_html(user, verification_url)

        email = EmailMultiAlternatives(
            subject=subject,
            body=plain_text,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email],
        )
        email.attach_alternative(html_body, 'text/html')
        email.send(fail_silently=False)
        logger.info(f"[Celery] Verification email sent to {user.email}")
        return True

    except User.DoesNotExist:
        logger.error(f"Verification email task: user #{user_id} not found.")
        return False
    except Exception as exc:
        logger.error(f"Verification email task failed for user #{user_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc)