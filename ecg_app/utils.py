import io
import os
import logging
from django.template.loader import render_to_string
from django.conf import settings

logger = logging.getLogger(__name__)


# ── Result image generator ─────────────────────────────────────────────────────

def generate_result_image(record):
    """
    Generates a PNG image of the ECG with a prediction result overlay panel.
    Saves the file to media/result_images/<year>/<month>/<day>/<id>/result.png
    Returns the absolute filesystem path (forward slashes), or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
        from PIL import Image

        # ── Load ECG image ────────────────────────────────────────────────────
        ecg_img = Image.open(record.image.path).convert('RGB')
        ecg_arr = np.array(ecg_img)

        # ── Colours ───────────────────────────────────────────────────────────
        colour_map = {
            'normal':   ('#10b981', '#d1fae5', 'Normal ECG'),
            'abnormal': ('#f59e0b', '#fef3c7', 'Abnormal Heartbeat'),
            'mi':       ('#ef4444', '#fee2e2', 'Myocardial Infarction'),
            'post_mi':  ('#0ea5e9', '#e0f2fe', 'Post MI History'),
        }
        cat = record.predicted_category
        accent, bg_hex, disp_label = colour_map.get(
            cat, ('#64748b', '#f1f5f9', record.get_predicted_category_display())
        )

        # ── Probability data ──────────────────────────────────────────────────
        prob_data = [
            ('Normal ECG',            record.normal_prob,   '#10b981'),
            ('Abnormal Heartbeat',     record.abnormal_prob, '#f59e0b'),
            ('Myocardial Infarction',  record.mi_prob,       '#ef4444'),
            ('Post MI History',        record.post_mi_prob,  '#0ea5e9'),
        ]

        # ── Figure: ECG top 60%, result panel bottom 35% ──────────────────────
        fig = plt.figure(figsize=(10, 8), facecolor='#0f172a')

        ax_ecg = fig.add_axes([0.02, 0.36, 0.96, 0.60])
        ax_ecg.imshow(ecg_arr)
        ax_ecg.axis('off')

        # Accent separator line
        ax_line = fig.add_axes([0.0, 0.345, 1.0, 0.009])
        ax_line.set_facecolor(accent)
        ax_line.axis('off')

        # Result panel
        ax = fig.add_axes([0.02, 0.01, 0.96, 0.33])
        ax.set_facecolor('#1e293b')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # ── Diagnosis badge (left) ────────────────────────────────────────────
        badge = FancyBboxPatch(
            (0.01, 0.54), 0.30, 0.40,
            boxstyle="round,pad=0.02",
            facecolor=bg_hex, edgecolor=accent, linewidth=2,
            transform=ax.transAxes, zorder=3
        )
        ax.add_patch(badge)

        ax.text(0.16, 0.82, disp_label,
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color='#0f172a', ha='center', va='center', zorder=4)

        ax.text(0.16, 0.65, f"Confidence: {record.confidence:.1f}%",
                transform=ax.transAxes, fontsize=8, color='#334155',
                ha='center', va='center', zorder=4)

        ax.text(0.16, 0.42, f"Record #{record.id}",
                transform=ax.transAxes, fontsize=7.5, color='#94a3b8',
                ha='center', va='center')

        ax.text(0.16, 0.28, record.upload_date.strftime("%Y-%m-%d %H:%M"),
                transform=ax.transAxes, fontsize=7, color='#64748b',
                ha='center', va='center')

        if record.patient:
            ax.text(0.16, 0.13, f"Patient: {record.patient.name}",
                    transform=ax.transAxes, fontsize=7, color='#64748b',
                    ha='center', va='center')

        # ── Probability bars (right) ──────────────────────────────────────────
        ax.text(0.37, 0.93, 'Probability Breakdown',
                transform=ax.transAxes, fontsize=7.5, color='#94a3b8',
                ha='left', va='center', style='italic')

        y_positions  = [0.72, 0.52, 0.32, 0.12]
        bar_x        = 0.37
        bar_max_w    = 0.58
        bar_h        = 0.13

        for (label, value, color), y in zip(prob_data, y_positions):
            fill_w = (value / 100.0) * bar_max_w

            # Track
            ax.add_patch(FancyBboxPatch(
                (bar_x, y - bar_h / 2), bar_max_w, bar_h,
                boxstyle="round,pad=0.005",
                facecolor='#334155', edgecolor='none',
                transform=ax.transAxes, zorder=2
            ))
            # Fill
            if fill_w > 0.005:
                ax.add_patch(FancyBboxPatch(
                    (bar_x, y - bar_h / 2), fill_w, bar_h,
                    boxstyle="round,pad=0.005",
                    facecolor=color, edgecolor='none', alpha=0.90,
                    transform=ax.transAxes, zorder=3
                ))

            ax.text(bar_x - 0.01, y, label,
                    transform=ax.transAxes, fontsize=6.8, color='#cbd5e1',
                    ha='right', va='center', zorder=4)

            ax.text(bar_x + bar_max_w + 0.01, y, f"{value:.1f}%",
                    transform=ax.transAxes, fontsize=7, color='#e2e8f0',
                    fontweight='bold', ha='left', va='center', zorder=4)

        ax.text(0.99, 0.06, 'CardioVision AI',
                transform=ax.transAxes, fontsize=7, color='#334155',
                ha='right', va='center', style='italic')

        # ── Save to disk ──────────────────────────────────────────────────────
        save_dir = os.path.join(
            settings.MEDIA_ROOT, 'result_images',
            record.upload_date.strftime('%Y/%m/%d'),
            str(record.id)
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'result.png')

        plt.savefig(save_path, format='png', dpi=150,
                    bbox_inches='tight', facecolor='#0f172a', edgecolor='none')
        plt.close(fig)

        # Return forward-slash path for xhtml2pdf (works on Windows too)
        return save_path.replace('\\', '/')

    except Exception as e:
        logger.error(f"Result image generation failed for ECG #{record.id}: {e}", exc_info=True)
        return None


# ── PDF generator ──────────────────────────────────────────────────────────────

def generate_pdf_report_content(ecg_id, request=None):
    """
    Generate a PDF report for an ECG record using xhtml2pdf.
    Returns bytes of the PDF, or None on failure.
    """
    try:
        from .models import ECGRecord
        from xhtml2pdf import pisa

        record = ECGRecord.objects.get(id=ecg_id)

        # ── Original ECG image path ───────────────────────────────────────────
        ecg_image_path = ''
        if record.image:
            ecg_image_path = record.image.path.replace('\\', '/')

        # ── Probabilities ─────────────────────────────────────────────────────
        probabilities = {
            'Normal ECG':            record.normal_prob,
            'Abnormal Heartbeat':    record.abnormal_prob,
            'Myocardial Infarction': record.mi_prob,
            'Post MI History':       record.post_mi_prob,
        }

        # ── LIME paths ────────────────────────────────────────────────────────
        def lime_path(cls_key):
            if not record.lime_images:
                return None
            url = record.lime_images.get(cls_key)
            if not url:
                return None
            media_url = settings.MEDIA_URL
            rel = url[len(media_url):] if url.startswith(media_url) else url.lstrip('/')
            abs_path = os.path.join(settings.MEDIA_ROOT, rel)
            return abs_path.replace('\\', '/') if os.path.exists(abs_path) else None

        lime_normal   = lime_path('normal')
        lime_abnormal = lime_path('abnormal')
        lime_mi       = lime_path('mi')
        lime_post_mi  = lime_path('post_mi')
        lime_ready    = any([lime_normal, lime_abnormal, lime_mi, lime_post_mi])

        # ── Generate (or reuse cached) result image ───────────────────────────
        result_image_path = None

        # Check if already saved from a previous call
        cached_dir  = os.path.join(
            settings.MEDIA_ROOT, 'result_images',
            record.upload_date.strftime('%Y/%m/%d'),
            str(record.id)
        )
        cached_path = os.path.join(cached_dir, 'result.png').replace('\\', '/')

        if os.path.exists(cached_path):
            result_image_path = cached_path
        else:
            result_image_path = generate_result_image(record)

        # ── Patient info ──────────────────────────────────────────────────────
        patient_name = record.patient.name if record.patient else None
        patient_age  = record.patient.age  if record.patient else None

        context = {
            'record':             record,
            'probabilities':      probabilities,
            'ecg_image_path':     ecg_image_path,
            'result_image_path':  result_image_path,   # ← NEW
            'lime_ready':         lime_ready,
            'lime_normal':        lime_normal,
            'lime_abnormal':      lime_abnormal,
            'lime_mi':            lime_mi,
            'lime_post_mi':       lime_post_mi,
            'patient_name':       patient_name,
            'patient_age':        patient_age,
        }

        html_string = render_to_string('ecg_app/pdf_report.html', context)

        pdf_buffer  = io.BytesIO()
        base_url    = str(settings.MEDIA_ROOT).replace('\\', '/')
        pisa_status = pisa.CreatePDF(html_string, dest=pdf_buffer, base_url=base_url)

        if pisa_status.err:
            logger.error(f"xhtml2pdf error for ECG #{ecg_id}: {pisa_status.err}")
            return None

        logger.info(f"PDF generated successfully for ECG #{ecg_id}")
        return pdf_buffer.getvalue()

    except Exception as e:
        logger.error(f"PDF generation error for ECG #{ecg_id}: {e}", exc_info=True)
        return None