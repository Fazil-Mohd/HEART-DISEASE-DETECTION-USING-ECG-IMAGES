import os
import logging
from celery import shared_task
from django.conf import settings
from .models import ECGRecord

logger = logging.getLogger(__name__)

@shared_task
def generate_lime_task(ecg_id):
    """
    Background task to generate 4 LIME overlays for a given ECG record.
    Provides immediate UI feedback on upload, while heavy LIME computation
    runs asynchronously.
    """
    try:
        record = ECGRecord.objects.get(id=ecg_id)
        
        # We need the ML model loaded in the worker process
        from .ml_model import ecg_model
        
        lime_dir = os.path.join(
            settings.MEDIA_ROOT, 'lime_explanations',
            record.upload_date.strftime('%Y/%m/%d'),
            str(record.id)
        )
        os.makedirs(lime_dir, exist_ok=True)
        
        # Generate overlays for all 4 classes
        results_by_class = ecg_model.generate_lime_explanation(record.image.path, lime_dir)
        
        # Build relative paths for the model JSON field
        lime_images = {}
        for cls_name, data in results_by_class.items():
            if data['image_name']:
                full_path = os.path.join(lime_dir, data['image_name'])
                rel_path = os.path.relpath(full_path, settings.MEDIA_ROOT).replace('\\', '/')
                lime_images[cls_name] = f"{settings.MEDIA_URL}{rel_path}"
            
        record.lime_images = lime_images
        record.lime_explanation_data = results_by_class
        record.save()
        
        logger.info(f"Successfully generated async LIME for ECG #{ecg_id}")
        return True
        
    except ECGRecord.DoesNotExist:
        logger.error(f"Generate LIME async task failed: ECG #{ecg_id} not found.")
        return False
    except Exception as e:
        logger.error(f"Generate LIME async task failed: {e}")
        return False
