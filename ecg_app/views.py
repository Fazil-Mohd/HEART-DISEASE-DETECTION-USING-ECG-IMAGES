# views.py - CORRECTED VERSION
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Count, Avg, Q
from django.contrib.admin.views.decorators import staff_member_required
import json
from datetime import timedelta
from django.utils import timezone
from django.contrib.auth.models import User

from .forms import UserRegisterForm, UserLoginForm, UserUpdateForm, ECGUploadForm, PatientForm
from .models import UserProfile, ECGRecord, Patient
from .ml_model import ecg_model
from django.views.decorators.csrf import csrf_exempt
import csv
import os
import re
from pathlib import Path
from django.conf import settings
from django.http import HttpResponse

# ========== AUTHENTICATION VIEWS ==========

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            # Create user but leave inactive until email is verified
            user = form.save(commit=False)
            user.is_active = False
            user.save()

            role = form.cleaned_data.get('role', 'user')
            UserProfile.objects.create(user=user, role=role)

            # Create verification token
            from .models import EmailVerificationToken
            token_obj = EmailVerificationToken.objects.create(user=user)

            # Build the absolute verification URL
            verification_url = request.build_absolute_uri(
                f'/verify-email/{token_obj.token}/'
            )

            # ── Try Celery first; fall back to synchronous send ──────────────
            email_sent = False
            try:
                from .tasks import send_verification_email_task
                send_verification_email_task.delay(user.id, verification_url)
                email_sent = True
            except Exception as celery_err:
                import logging
                logging.getLogger(__name__).warning(
                    f"Celery unavailable ({celery_err}), falling back to sync email."
                )

            if not email_sent:
                # Synchronous fallback — sends inline without Celery
                from .tasks import _send_verification_email_sync
                try:
                    _send_verification_email_sync(user, verification_url)
                    email_sent = True
                except Exception as mail_err:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Sync verification email also failed: {mail_err}"
                    )

            # Store the email in session for display on the pending page
            request.session['pending_verification_email'] = user.email

            messages.success(
                request,
                f'Account created! A verification link has been sent to {user.email}. '
                f'Please check your inbox (and spam folder) to activate your account.'
            )
            return redirect('email_pending')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegisterForm()

    return render(request, 'ecg_app/auth/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user_candidate = authenticate(username=username, password=password)

        if user_candidate is not None:
            if not user_candidate.is_active:
                # Account exists but email not yet verified
                from .models import EmailVerificationToken
                try:
                    token_obj = user_candidate.email_verification
                    has_pending_token = not token_obj.is_verified
                except EmailVerificationToken.DoesNotExist:
                    has_pending_token = False

                if has_pending_token:
                    request.session['pending_verification_email'] = user_candidate.email
                    messages.warning(
                        request,
                        'Your email address has not been verified yet. '
                        'Please check your inbox for the verification link, '
                        'or resend it below.'
                    )
                    return render(request, 'ecg_app/auth/login.html', {
                        'form': form,
                        'show_resend': True,
                        'unverified_username': username,
                    })
                else:
                    messages.error(request, 'Your account is inactive. Please contact support.')
            else:
                login(request, user_candidate)
                messages.success(request, f'Welcome back, {user_candidate.username}!')
                next_page = request.GET.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = UserLoginForm()

    return render(request, 'ecg_app/auth/login.html', {'form': form})



@login_required
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('home')


# ========== EMAIL VERIFICATION VIEWS ==========

def verify_email_view(request, token):
    """
    Handles the verification link clicked by the user from their email.
    Validates the token, activates the account, and logs the user in.
    """
    from .models import EmailVerificationToken

    try:
        token_obj = EmailVerificationToken.objects.select_related('user').get(token=token)
    except EmailVerificationToken.DoesNotExist:
        messages.error(
            request,
            'This verification link is invalid or has already been used. '
            'Please register again or contact support.'
        )
        return render(request, 'ecg_app/auth/email_verified.html', {
            'success': False,
            'error_type': 'invalid',
        })

    if token_obj.is_verified:
        messages.info(request, 'Your email has already been verified. You can log in.')
        return redirect('login')

    if token_obj.is_expired():
        messages.warning(
            request,
            'This verification link has expired (links are valid for 24 hours). '
            'Please request a new one.'
        )
        request.session['pending_verification_email'] = token_obj.user.email
        return render(request, 'ecg_app/auth/email_verified.html', {
            'success': False,
            'error_type': 'expired',
            'user': token_obj.user,
        })

    # ── Everything is valid — activate the account ───────────────────────────
    user = token_obj.user
    user.is_active = True
    user.save(update_fields=['is_active'])

    token_obj.is_verified = True
    token_obj.save(update_fields=['is_verified'])

    login(request, user)
    messages.success(
        request,
        f'🎉 Welcome to CardioVision AI, {user.first_name or user.username}! '
        f'Your email has been verified and your account is now active.'
    )
    return render(request, 'ecg_app/auth/email_verified.html', {
        'success': True,
        'user': user,
    })


def email_verification_pending_view(request):
    """
    Shown after registration: tells the user to check their inbox.
    """
    email = request.session.get('pending_verification_email', '')
    return render(request, 'ecg_app/auth/email_pending.html', {'email': email})


def resend_verification_view(request):
    """
    Resends the verification email for an unverified account.
    Accepts POST with 'username' field or pulls from session.
    """
    from .models import EmailVerificationToken

    if request.method != 'POST':
        return redirect('login')

    username = request.POST.get('username', '').strip()
    if not username:
        messages.error(request, 'Please provide your username to resend the verification email.')
        return redirect('login')

    try:
        user = User.objects.get(username=username, is_active=False)
    except User.DoesNotExist:
        # Don't reveal whether the user exists — show the same success message
        messages.success(
            request,
            'If that account exists and is unverified, '
            'a new verification email has been sent.'
        )
        return redirect('email_pending')

    # Refresh or create the token
    token_obj, _ = EmailVerificationToken.objects.get_or_create(user=user)
    if token_obj.is_verified:
        messages.info(request, 'This account is already verified. Please log in.')
        return redirect('login')

    # Generate a fresh token if the old one is expired
    if token_obj.is_expired():
        token_obj.delete()
        token_obj = EmailVerificationToken.objects.create(user=user)

    verification_url = request.build_absolute_uri(
        f'/verify-email/{token_obj.token}/'
    )

    # ── Try Celery first; fall back to synchronous send ──────────────────────
    email_sent = False
    try:
        from .tasks import send_verification_email_task
        send_verification_email_task.delay(user.id, verification_url)
        email_sent = True
    except Exception:
        pass

    if not email_sent:
        from .tasks import _send_verification_email_sync
        try:
            _send_verification_email_sync(user, verification_url)
            email_sent = True
        except Exception as mail_err:
            import logging
            logging.getLogger(__name__).error(f"Resend sync email failed: {mail_err}")

    request.session['pending_verification_email'] = user.email
    messages.success(
        request,
        f'A fresh verification link has been sent to {user.email}. '
        f'Please check your inbox and spam folder.'
    )
    return redirect('email_pending')


# ========== CORE USER VIEWS ==========

def home_view(request):
    if request.user.is_authenticated:
        recent_ecgs = ECGRecord.objects.filter(user=request.user)[:3]
        return render(request, 'ecg_app/home.html', {
            'recent_ecgs': recent_ecgs,
            'user': request.user
        })
    return render(request, 'ecg_app/home.html')


@login_required
def dashboard_view(request):
    user = request.user
    user_ecgs = ECGRecord.objects.filter(user=user)
    total_ecgs = user_ecgs.count()

    completed_ecgs  = user_ecgs.filter(status='completed').count()
    processing_ecgs = user_ecgs.filter(status='processing').count()
    failed_ecgs     = user_ecgs.filter(status='failed').count()

    normal_ecgs   = user_ecgs.filter(predicted_category='normal',   status='completed').count()
    abnormal_ecgs = user_ecgs.filter(predicted_category='abnormal', status='completed').count()
    mi_ecgs       = user_ecgs.filter(predicted_category='mi',       status='completed').count()
    post_mi_ecgs  = user_ecgs.filter(predicted_category='post_mi',  status='completed').count()

    avg_confidence = user_ecgs.filter(status='completed').aggregate(
        avg_conf=Avg('confidence')
    )['avg_conf'] or 0

    latest_ecg    = user_ecgs.order_by('-upload_date').first()
    latest_ecg_id = latest_ecg.id if latest_ecg else None
    recent_ecgs   = user_ecgs.order_by('-upload_date')[:10]

    today = timezone.now().date()
    recent_activity = []
    for i in range(6, -1, -1):
        date  = today - timedelta(days=i)
        count = user_ecgs.filter(upload_date__date=date).count()
        max_count = max(1, user_ecgs.filter(
            upload_date__date__gte=today - timedelta(days=6)
        ).count())
        height = int((count / max(1, max_count)) * 100) + 20
        recent_activity.append({
            'date': date.strftime('%a'),
            'count': count,
            'height': min(height, 120)
        })

    context = {
        'total_ecgs':       total_ecgs,
        'completed_ecgs':   completed_ecgs,
        'processing_ecgs':  processing_ecgs,
        'failed_ecgs':      failed_ecgs,
        'normal_ecgs':      normal_ecgs,
        'abnormal_ecgs':    abnormal_ecgs,
        'mi_ecgs':          mi_ecgs,
        'post_mi_ecgs':     post_mi_ecgs,
        'avg_confidence':   avg_confidence,
        'latest_ecg_id':    latest_ecg_id,
        'recent_ecgs':      recent_ecgs,
        'recent_activity':  recent_activity,
        'user':             user,
    }
    return render(request, 'ecg_app/dashboard.html', context)


@login_required
def profile_view(request):
    user      = request.user
    user_ecgs = ECGRecord.objects.filter(user=user)
    total_ecgs    = user_ecgs.count()
    normal_ecgs   = user_ecgs.filter(predicted_category='normal', status='completed').count()
    abnormal_ecgs = user_ecgs.filter(status='completed').exclude(predicted_category='normal').count()
    success_rate  = round((normal_ecgs / total_ecgs) * 100, 1) if total_ecgs > 0 else 0
    recent_ecgs   = user_ecgs.order_by('-upload_date')[:5]

    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        user_form = UserUpdateForm(instance=request.user)

    context = {
        'user_form':    user_form,
        'user':         user,
        'total_ecgs':   total_ecgs,
        'normal_ecgs':  normal_ecgs,
        'abnormal_ecgs': abnormal_ecgs,
        'success_rate': success_rate,
        'recent_ecgs':  recent_ecgs,
    }
    return render(request, 'ecg_app/profile.html', context)


@login_required
def upload_ecg_view(request):
    """Upload ECG for analysis"""
    recent_ecgs = (
        ECGRecord.objects.filter(user=request.user).order_by('-upload_date')[:3]
        if request.user.is_authenticated else []
    )

    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            ecg_record        = form.save(commit=False)
            ecg_record.user   = request.user
            ecg_record.status = 'processing'
            ecg_record.save()

            try:
                result = ecg_model.predict(ecg_record.image.path)

                if result:
                    ecg_record.predicted_category = result['predicted_class']
                    ecg_record.confidence         = result['confidence'] * 100
                    ecg_record.status             = 'completed'

                    all_probs = result.get('all_probabilities', {})
                    ecg_record.normal_prob   = all_probs.get('normal',   0) * 100
                    ecg_record.abnormal_prob = all_probs.get('abnormal', 0) * 100
                    ecg_record.mi_prob       = all_probs.get('mi',       0) * 100
                    ecg_record.post_mi_prob  = all_probs.get('post_mi',  0) * 100
                    ecg_record.save()

                    # ── Fire LIME and email as INDEPENDENT parallel tasks ──────
                    # Email sends immediately (~5 s) without waiting 60 s for LIME
                    from .tasks import generate_lime_task, send_pdf_report_email_task
                    try:
                        generate_lime_task.delay(ecg_record.id)
                        send_pdf_report_email_task.delay(ecg_record.id)
                    except Exception as celery_err:
                        print(f"Warning: Could not queue background tasks (is Redis running?): {celery_err}")
                    # ──────────────────────────────────────────────────────────

                    messages.success(
                        request,
                        f'Analysis completed! Confidence: {ecg_record.confidence:.1f}%. '
                        f'A PDF report will be emailed to you shortly.'
                    )
                    return redirect('ecg_result', ecg_id=ecg_record.id)

                else:
                    ecg_record.status = 'failed'
                    ecg_record.save()
                    messages.error(request, 'Failed to analyze ECG. Please try again.')
                    return redirect('upload')

            except Exception as e:
                ecg_record.status = 'failed'
                ecg_record.save()
                messages.error(request, f'Error processing image: {str(e)}')
                return redirect('upload')

        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = ECGUploadForm(user=request.user)

    return render(request, 'ecg_app/upload.html', {
        'form': form,
        'recent_ecgs': recent_ecgs
    })


@login_required
def ecg_result_view(request, ecg_id):
    ecg_record = get_object_or_404(ECGRecord, id=ecg_id, user=request.user)

    previous_ecg = None
    trend_points = []

    if ecg_record.patient:
        previous_ecg = ECGRecord.objects.filter(
            patient=ecg_record.patient, 
            upload_date__lt=ecg_record.upload_date,
            status='completed'
        ).order_by('-upload_date').first()

        if previous_ecg:
            prev_cat = previous_ecg.get_predicted_category_display()
            curr_cat = ecg_record.get_predicted_category_display()
            prev_date = previous_ecg.upload_date.strftime('%b %d, %Y')
            
            prev_conf = f"{previous_ecg.confidence:.1f}%"
            curr_conf = f"{ecg_record.confidence:.1f}%"
            
            trend_points = [
                {'icon': 'fas fa-angle-right text-primary', 'label': f'Prior Exam ({prev_date})', 'value': f'{prev_cat} ({prev_conf})'},
                {'icon': 'fas fa-angle-right text-primary', 'label': 'Current Exam', 'value': f'{curr_cat} ({curr_conf})'},
            ]
            
            if prev_cat == curr_cat:
                trend_points.append({'icon': 'fas fa-check-circle text-success', 'label': 'Status', 'value': 'Stable'})
                trend_points.append({'icon': 'fas fa-check-circle text-success', 'label': 'Note', 'value': 'No significant changes in diagnosis since last evaluation.'})
            elif prev_cat == 'Normal ECG' and curr_cat != 'Normal ECG':
                trend_points.append({'icon': 'fas fa-exclamation-circle text-warning', 'label': 'Status', 'value': 'New Finding Detected'})
                trend_points.append({'icon': 'fas fa-exclamation-circle text-warning', 'label': 'Action', 'value': 'Consult with a healthcare professional to review these new changes.'})
            elif prev_cat != 'Normal ECG' and curr_cat == 'Normal ECG':
                trend_points.append({'icon': 'fas fa-check-circle text-success', 'label': 'Status', 'value': 'Improvement'})
                trend_points.append({'icon': 'fas fa-check-circle text-success', 'label': 'Note', 'value': 'No significant abnormalities detected in the current scan.'})
            else:
                trend_points.append({'icon': 'fas fa-exclamation-circle text-warning', 'label': 'Status', 'value': 'Diagnosis Shift Over Time'})
                trend_points.append({'icon': 'fas fa-exclamation-circle text-warning', 'label': 'Note', 'value': 'A detailed clinical correlation is recommended to understand progression.'})

    probabilities = {
        'Normal ECG':            ecg_record.normal_prob,
        'Abnormal Heartbeat':    ecg_record.abnormal_prob,
        'Myocardial Infarction': ecg_record.mi_prob,
        'Post MI History':       ecg_record.post_mi_prob,
    }

    context = {
        'record':        ecg_record,
        'probabilities': probabilities,
        'previous_ecg':  previous_ecg,
        'trend_points':  trend_points,
    }
    return render(request, 'ecg_app/results.html', context)


@login_required
def ecg_history_view(request):
    ecg_records = ECGRecord.objects.filter(user=request.user).order_by('-upload_date')

    status_filter = request.GET.get('status', 'all')
    if status_filter != 'all':
        ecg_records = ecg_records.filter(status=status_filter)

    category_filter = request.GET.get('category', 'all')
    if category_filter != 'all':
        ecg_records = ecg_records.filter(predicted_category=category_filter)

    start_date = request.GET.get('start_date')
    end_date   = request.GET.get('end_date')
    if start_date:
        ecg_records = ecg_records.filter(upload_date__date__gte=start_date)
    if end_date:
        ecg_records = ecg_records.filter(upload_date__date__lte=end_date)

    total_records    = ecg_records.count()
    completed_count  = ECGRecord.objects.filter(user=request.user, status='completed').count()
    processing_count = ECGRecord.objects.filter(user=request.user, status='processing').count()
    failed_count     = ECGRecord.objects.filter(user=request.user, status='failed').count()

    avg_confidence = ECGRecord.objects.filter(
        user=request.user, status='completed', confidence__isnull=False
    ).aggregate(avg_conf=Avg('confidence'))['avg_conf'] or 0

    category_counts = {}
    category_data = ECGRecord.objects.filter(
        user=request.user, predicted_category__isnull=False
    ).values('predicted_category').annotate(count=Count('predicted_category'))
    for item in category_data:
        category_counts[item['predicted_category']] = item['count']

    most_common = ECGRecord.objects.filter(
        user=request.user, predicted_category__isnull=False
    ).values('predicted_category').annotate(
        count=Count('predicted_category')
    ).order_by('-count').first()
    most_common_category = most_common['predicted_category'] if most_common else None

    this_month = timezone.now().replace(day=1)
    this_month_count = ECGRecord.objects.filter(
        user=request.user, upload_date__gte=this_month
    ).count()

    today = timezone.now().date()
    recent_activity = []
    base_queryset = ECGRecord.objects.filter(user=request.user)
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        count = base_queryset.filter(upload_date__date=date).count()
        if i == 0:
            label = 'Today'
        elif i == 1:
            label = 'Yesterday'
        else:
            label = date.strftime('%b %d')
        recent_activity.append({
            'date': label,
            'count': count
        })

    paginator   = Paginator(ecg_records, 10)
    page_number = request.GET.get('page')
    page_obj    = paginator.get_page(page_number)

    context = {
        'page_obj':             page_obj,
        'total_records':        total_records,
        'completed_count':      completed_count,
        'processing_count':     processing_count,
        'failed_count':         failed_count,
        'avg_confidence':       avg_confidence,
        'categories':           ECGRecord.CATEGORY_CHOICES,
        'category_counts':      category_counts,
        'most_common_category': most_common_category,
        'this_month_count':     this_month_count,
        'recent_activity_json': json.dumps(recent_activity),
    }
    return render(request, 'ecg_app/history.html', context)


# ========== PATIENT MANAGEMENT VIEWS ==========

@login_required
def patient_list_view(request):
    patients = Patient.objects.filter(user=request.user)
    return render(request, 'ecg_app/patients/patient_list.html', {'patients': patients})


@login_required
def patient_detail_view(request, patient_id):
    patient     = get_object_or_404(Patient, id=patient_id, user=request.user)
    ecg_records = patient.ecg_records.order_by('-upload_date')
    return render(request, 'ecg_app/patients/patient_detail.html', {
        'patient':     patient,
        'ecg_records': ecg_records,
    })


@login_required
def patient_create_view(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient      = form.save(commit=False)
            patient.user = request.user
            patient.save()
            messages.success(request, f'Patient {patient.name} added successfully.')
            return redirect('patient_list')
    else:
        form = PatientForm()

    return render(request, 'ecg_app/patients/patient_form.html', {
        'form':        form,
        'title':       'Add New Patient',
        'button_text': 'Add Patient',
    })


@login_required
def patient_update_view(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)

    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            patient = form.save()
            messages.success(request, f'Patient {patient.name} updated successfully.')
            return redirect('patient_detail', patient_id=patient.id)
    else:
        form = PatientForm(instance=patient)

    return render(request, 'ecg_app/patients/patient_form.html', {
        'form':        form,
        'title':       'Edit Patient',
        'button_text': 'Update Patient',
        'patient':     patient,
    })


@login_required
def patient_delete_view(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id, user=request.user)

    if request.method == 'POST':
        name = patient.name
        patient.delete()
        messages.success(request, f'Patient {name} has been removed.')
        return redirect('patient_list')

    return render(request, 'ecg_app/patients/patient_confirm_delete.html', {'patient': patient})


# ========== API VIEWS ==========

@login_required
@csrf_exempt
def api_train_model(request):
    if request.method == 'POST':
        try:
            success = ecg_model.train_model()
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Model trained successfully',
                    'model_info': ecg_model.get_model_info()
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Training failed'}, status=500)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'error': 'Only POST allowed'}, status=405)


@login_required
def api_user_stats(request):
    user_ecgs  = ECGRecord.objects.filter(user=request.user)
    total_ecgs = user_ecgs.count()

    category_distribution = {}
    category_data = user_ecgs.filter(
        status='completed', predicted_category__isnull=False
    ).values('predicted_category').annotate(count=Count('predicted_category'))
    for item in category_data:
        category_distribution[item['predicted_category']] = item['count']

    normal_ecgs   = category_distribution.get('normal', 0)
    abnormal_ecgs = total_ecgs - normal_ecgs
    success_rate  = round((normal_ecgs / total_ecgs) * 100, 1) if total_ecgs > 0 else 0

    return JsonResponse({
        'total_ecgs':             total_ecgs,
        'username':               request.user.username,
        'category_distribution':  category_distribution,
        'normal_ecgs':            normal_ecgs,
        'abnormal_ecgs':          abnormal_ecgs,
        'success_rate':           success_rate,
    })


@login_required
def api_lime_explanation(request, ecg_id):
    record   = get_object_or_404(ECGRecord, id=ecg_id, user=request.user)
    is_ready = bool(record.lime_images and len(record.lime_images) > 0)

    print(f"[DEBUG api_lime] ID: {ecg_id}, is_ready: {is_ready}")
    print(f"[DEBUG api_lime] type: {type(record.lime_images)}, content: {record.lime_images}")

    return JsonResponse({
        'ecg_id':         ecg_id,
        'is_ready':       is_ready,
        'lime_images':    record.lime_images or {},
        'lime_data':      record.lime_explanation_data or {},
        'class_names':    ecg_model.class_names,
        'predicted_class': record.predicted_category,
    })


@login_required
def generate_pdf_report(request, ecg_id):
    """
    Serve a PDF report inline in the browser (also downloadable).
    """
    from .utils import generate_pdf_report_content
    from django.http import Http404

    get_object_or_404(ECGRecord, id=ecg_id, user=request.user)

    pdf_bytes = generate_pdf_report_content(ecg_id, request)
    if pdf_bytes is None:
        raise Http404("Report could not be generated.")

    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="ECG_Report_{ecg_id}.pdf"'
    return response


# ========== ADMIN VIEWS ==========

@staff_member_required
def admin_dashboard_view(request):
    total_users = User.objects.count()
    total_ecgs  = ECGRecord.objects.count()

    today = timezone.now().date()
    active_users_today = User.objects.filter(
        Q(last_login__date=today) | Q(date_joined__date=today)
    ).distinct().count()

    recent_users = User.objects.annotate(
        ecg_count=Count('ecg_records')
    ).order_by('-date_joined')[:10]

    recent_ecgs = ECGRecord.objects.select_related('user').order_by('-upload_date')[:10]

    active_percentage = int((active_users_today / total_users * 100)) if total_users > 0 else 0

    ecgs_today     = ECGRecord.objects.filter(upload_date__date=today).count()
    new_users_today = User.objects.filter(date_joined__date=today).count()

    context = {
        'total_users':         total_users,
        'total_ecgs':          total_ecgs,
        'active_users_today':  active_users_today,
        'active_percentage':   active_percentage,
        'recent_users':        recent_users,
        'recent_ecgs':         recent_ecgs,
        'user_growth':         12,
        'ecg_growth':          8,
        'avg_growth':          5,
        'growth_rate':         15,
        'ecgs_today':          ecgs_today,
        'new_users_today':     new_users_today,
    }
    return render(request, 'ecg_app/admin_dashboard.html', context)


def admin_login_view(request):
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('admin_dashboard')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user     = authenticate(username=username, password=password)

        if user is not None and user.is_superuser:
            login(request, user)
            messages.success(request, 'Welcome to Admin Dashboard!')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid admin credentials or insufficient permissions.')

    return render(request, 'ecg_app/auth/admin_login.html')


@login_required
def export_history_csv_view(request):
    ecg_records = ECGRecord.objects.filter(user=request.user).order_by('-upload_date')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="ecg_history.csv"'

    writer = csv.writer(response)
    writer.writerow(['ID', 'Date', 'Prediction', 'Confidence', 'Status', 'Patient', 'Notes'])

    for ecg in ecg_records:
        writer.writerow([
            ecg.id,
            ecg.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            ecg.get_predicted_category_display(),
            f"{ecg.confidence:.1f}%" if ecg.confidence else "",
            ecg.status,
            ecg.patient.name if ecg.patient else "",
            ecg.notes or "",
        ])

    return response


# ========== MODEL PERFORMANCE DASHBOARD VIEW ==========

@login_required
def model_performance_view(request):
    """
    Renders the ECG Classification Model Performance Dashboard.
    Parses training CSV logs and classification report to pass structured
    data to the template for Chart.js visualisations.
    """
    base_dir = settings.BASE_DIR
    model_dir = base_dir / 'resnet_models'

    # ── Parse training CSV logs ─────────────────────────────────────────────
    def parse_training_csv(csv_path):
        rows = []
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    rows.append({
                        'epoch':        i + 1,
                        'accuracy':     round(float(row.get('accuracy', 0)), 4),
                        'val_accuracy': round(float(row.get('val_accuracy', 0)), 4),
                        'loss':         round(float(row.get('loss', 0)), 4),
                        'val_loss':     round(float(row.get('val_loss', 0)), 4),
                    })
        except Exception:
            pass
        return rows

    phase1_data = parse_training_csv(model_dir / 'phase1_log.csv')
    phase2_data = parse_training_csv(model_dir / 'phase2_log.csv')

    # ── Parse classification_report.txt ─────────────────────────────────────
    class_metrics = {}   # {class_name: {precision, recall, f1, support}}
    overall_accuracy = None
    macro_precision = macro_recall = macro_f1 = None
    val_accuracy_txt = val_loss_txt = None

    report_path = model_dir / 'classification_report.txt'
    try:
        with open(report_path, encoding='utf-8') as f:
            content = f.read()

        # Extract Validation Accuracy / Loss from header lines
        for line in content.splitlines():
            if 'Val Accuracy' in line or 'Validation Acc' in line:
                m = re.search(r'[\d.]+', line.split(':')[1] if ':' in line else line)
                if m:
                    val_accuracy_txt = float(m.group())
            elif 'Val Loss' in line or 'Validation Loss' in line:
                m = re.search(r'[\d.]+', line.split(':')[1] if ':' in line else line)
                if m:
                    val_loss_txt = float(m.group())

        # Parse per-class rows: "  classname   0.xxxx   0.xxxx   0.xxxx   NNN"
        class_name_pattern = re.compile(
            r'^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$'
        )
        for line in content.splitlines():
            m = class_name_pattern.match(line)
            if m:
                label = m.group(1).strip()
                prec  = float(m.group(2))
                rec   = float(m.group(3))
                f1    = float(m.group(4))
                supp  = int(m.group(5))
                if label == 'accuracy':
                    overall_accuracy = prec  # for accuracy row, col2 is the score
                elif label == 'macro':
                    # 'macro avg' line split differently — skip, handled below
                    pass
                else:
                    class_metrics[label] = {
                        'precision': prec,
                        'recall':    rec,
                        'f1':        f1,
                        'support':   supp,
                    }

        # 'accuracy' row has only 3 values
        acc_match = re.search(r'accuracy\s+([\d.]+)\s+(\d+)', content)
        if acc_match:
            overall_accuracy = float(acc_match.group(1))

        # macro avg row
        macro_match = re.search(
            r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)', content
        )
        if macro_match:
            macro_precision = float(macro_match.group(1))
            macro_recall    = float(macro_match.group(2))
            macro_f1        = float(macro_match.group(3))
    except Exception:
        pass

    # ── Derived insights ────────────────────────────────────────────────────
    best_class  = max(class_metrics, key=lambda c: class_metrics[c]['f1'],  default=None) if class_metrics else None
    worst_class = min(class_metrics, key=lambda c: class_metrics[c]['f1'],  default=None) if class_metrics else None

    # Overfitting detection: compare final train vs val accuracy in phase2
    overfit_status = 'unknown'
    overfit_detail = ''
    if phase2_data:
        last = phase2_data[-1]
        gap  = last['accuracy'] - last['val_accuracy']
        if gap > 0.12:
            overfit_status = 'overfitting'
            overfit_detail = f"Train accuracy ({last['accuracy']*100:.1f}%) is significantly higher than validation accuracy ({last['val_accuracy']*100:.1f}%), suggesting overfitting."
        elif gap < -0.05:
            overfit_status = 'underfitting'
            overfit_detail = f"Validation accuracy ({last['val_accuracy']*100:.1f}%) exceeds training accuracy — the model may be underfitting."
        else:
            overfit_status = 'good_fit'
            overfit_detail = f"Training and validation accuracies are well-aligned ({last['accuracy']*100:.1f}% vs {last['val_accuracy']*100:.1f}%), indicating a good fit."

    # Display names mapping
    display_names = {
        'normal':   'Normal ECG',
        'abnormal': 'Abnormal Heartbeat',
        'mi':       'Myocardial Infarction',
        'post_mi':  'Post MI History',
    }
    short_names = {
        'normal':   'Normal ECG',
        'abnormal': 'Abnormal HB',
        'mi':       'Myocardial Inf.',
        'post_mi':  'Post MI',
    }

    # Pre-compute percentage values for mini-bars (avoids broken floatformat in template)
    for cls_key, m in class_metrics.items():
        m['precision_pct'] = round(m['precision'] * 100, 1)
        m['recall_pct']    = round(m['recall']    * 100, 1)
        m['f1_pct']        = round(m['f1']        * 100, 1)

    # Pre-build per-class Chart.js JSON (eliminates Django template loops in <script>)
    per_class_chart = {
        'labels':    [short_names.get(c, c) for c in class_metrics],
        'precision': [class_metrics[c]['precision'] for c in class_metrics],
        'recall':    [class_metrics[c]['recall']    for c in class_metrics],
        'f1':        [class_metrics[c]['f1']        for c in class_metrics],
    }

    context = {
        # Overall metrics
        'overall_accuracy':    round((val_accuracy_txt or overall_accuracy or 0) * 100, 2),
        'val_loss':            round(val_loss_txt or 0, 4),
        'macro_precision':     round((macro_precision or 0) * 100, 2),
        'macro_recall':        round((macro_recall or 0) * 100, 2),
        'macro_f1':            round((macro_f1 or 0) * 100, 2),

        # Per-class
        'class_metrics':       class_metrics,
        'display_names':       display_names,

        # Training data as JSON strings for Chart.js
        'phase1_json':          json.dumps(phase1_data),
        'phase2_json':          json.dumps(phase2_data),
        # Per-class chart JSON (no Django tags needed in <script>)
        'per_class_chart_json': json.dumps(per_class_chart),

        # Insights
        'best_class':          best_class,
        'best_class_display':  display_names.get(best_class, best_class) if best_class else '',
        'best_class_f1':       round(class_metrics[best_class]['f1'] * 100, 1) if best_class else 0,
        'worst_class':         worst_class,
        'worst_class_display': display_names.get(worst_class, worst_class) if worst_class else '',
        'worst_class_f1':      round(class_metrics[worst_class]['f1'] * 100, 1) if worst_class else 0,
        'overfit_status':      overfit_status,
        'overfit_detail':      overfit_detail,

        # Phase summary counts
        'phase1_epochs':       len(phase1_data),
        'phase2_epochs':       len(phase2_data),
    }
    return render(request, 'ecg_app/model_performance.html', context)