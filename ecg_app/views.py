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

from .forms import UserRegisterForm, UserLoginForm, UserUpdateForm, ECGUploadForm
from .models import UserProfile, ECGRecord
from .ml_model import ecg_model
from django.views.decorators.csrf import csrf_exempt
import csv
import os
from django.http import HttpResponse

# ========== AUTHENTICATION VIEWS ==========

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Create user profile
            UserProfile.objects.create(user=user)
            
            # Log the user in
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to ECG Analyzer.')
            return redirect('dashboard')
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
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                
                # Redirect to next page if exists
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

# ========== CORE USER VIEWS ==========

def home_view(request):
    """Home page view"""
    if request.user.is_authenticated:
        recent_ecgs = ECGRecord.objects.filter(user=request.user)[:3]
        return render(request, 'ecg_app/home.html', {
            'recent_ecgs': recent_ecgs,
            'user': request.user
        })
    return render(request, 'ecg_app/home.html')

@login_required
def dashboard_view(request):
    """Main dashboard view"""
    user = request.user
    
    # Get user's ECG stats
    user_ecgs = ECGRecord.objects.filter(user=user)
    total_ecgs = user_ecgs.count()
    
    # Get counts by status
    completed_ecgs = user_ecgs.filter(status='completed').count()
    processing_ecgs = user_ecgs.filter(status='processing').count()
    failed_ecgs = user_ecgs.filter(status='failed').count()
    
    # Get counts by category
    normal_ecgs = user_ecgs.filter(predicted_category='normal', status='completed').count()
    abnormal_ecgs = user_ecgs.filter(predicted_category='abnormal', status='completed').count()
    mi_ecgs = user_ecgs.filter(predicted_category='mi', status='completed').count()
    post_mi_ecgs = user_ecgs.filter(predicted_category='post_mi', status='completed').count()
    
    # Calculate average confidence for completed analyses
    avg_confidence = user_ecgs.filter(status='completed').aggregate(
        avg_conf=Avg('confidence')
    )['avg_conf'] or 0
    
    # Get latest ECG ID for "View Last Result" button
    latest_ecg = user_ecgs.order_by('-upload_date').first()
    latest_ecg_id = latest_ecg.id if latest_ecg else None
    
    # Get recent ECGs (last 10)
    recent_ecgs = user_ecgs.order_by('-upload_date')[:10]
    
    # Get recent activity (last 7 days)
    today = timezone.now().date()
    recent_activity = []
    
    for i in range(6, -1, -1):  # Last 7 days including today
        date = today - timedelta(days=i)
        count = user_ecgs.filter(
            upload_date__date=date
        ).count()
        
        # Calculate height for chart visualization
        max_count = max(1, user_ecgs.filter(
            upload_date__date__gte=today - timedelta(days=6)
        ).count())
        height = int((count / max(1, max_count)) * 100) + 20  # 20-120px
        
        recent_activity.append({
            'date': date.strftime('%a'),  # Short day name
            'count': count,
            'height': min(height, 120)  # Cap at 120px
        })
    
    context = {
        'total_ecgs': total_ecgs,
        'completed_ecgs': completed_ecgs,
        'processing_ecgs': processing_ecgs,
        'failed_ecgs': failed_ecgs,
        'normal_ecgs': normal_ecgs,
        'abnormal_ecgs': abnormal_ecgs,
        'mi_ecgs': mi_ecgs,
        'post_mi_ecgs': post_mi_ecgs,
        'avg_confidence': avg_confidence,
        'latest_ecg_id': latest_ecg_id,
        'recent_ecgs': recent_ecgs,
        'recent_activity': recent_activity,
        'user': user,
    }
    
    return render(request, 'ecg_app/dashboard.html', context)

@login_required
def profile_view(request):
    """User profile view"""
    user = request.user
    
    # Get user statistics
    user_ecgs = ECGRecord.objects.filter(user=user)
    total_ecgs = user_ecgs.count()
    
    # Get category counts
    normal_ecgs = user_ecgs.filter(predicted_category='normal', status='completed').count()
    abnormal_ecgs = user_ecgs.filter(status='completed').exclude(predicted_category='normal').count()
    
    # Calculate success rate (percentage of normal results)
    if total_ecgs > 0:
        success_rate = round((normal_ecgs / total_ecgs) * 100, 1)
    else:
        success_rate = 0
    
    # Get recent ECGs for activity timeline
    recent_ecgs = user_ecgs.order_by('-upload_date')[:5]
    
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        
        if user_form.is_valid():
            user_form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        user_form = UserUpdateForm(instance=request.user)
    
    context = {
        'user_form': user_form,
        'user': user,
        'total_ecgs': total_ecgs,
        'normal_ecgs': normal_ecgs,
        'abnormal_ecgs': abnormal_ecgs,
        'success_rate': success_rate,
        'recent_ecgs': recent_ecgs,
    }
    
    return render(request, 'ecg_app/profile.html', context)

@login_required
def upload_ecg_view(request):
    """Upload ECG for analysis"""
    recent_ecgs = ECGRecord.objects.filter(user=request.user).order_by('-upload_date')[:3] if request.user.is_authenticated else []
    
    if request.method == 'POST':
        form = ECGUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ecg_record = form.save(commit=False)
            ecg_record.user = request.user
            ecg_record.status = 'processing'
            
            ecg_record.save()
            
            try:
                # Make prediction using the model
                result = ecg_model.predict(ecg_record.image.path)
                
                if result:
                    # Update record with prediction
                    ecg_record.predicted_category = result['predicted_class']
                    ecg_record.confidence = result['confidence'] * 100
                    ecg_record.status = 'completed'
                    
                    # Store probabilities as JSON
                    all_probs = result.get('all_probabilities', {})

                    ecg_record.normal_prob = all_probs.get('normal', 0) * 100
                    ecg_record.abnormal_prob = all_probs.get('abnormal', 0) * 100
                    ecg_record.mi_prob = all_probs.get('mi', 0) * 100
                    ecg_record.post_mi_prob = all_probs.get('post_mi', 0) * 100
                    
                    ecg_record.save()
                    
                    # ── LIME Explainability (Async Celery Task) ─────────────────
                    # Offload to Celery so the user doesn't wait 30 seconds
                    from .tasks import generate_lime_task
                    generate_lime_task.delay(ecg_record.id)
                    # ────────────────────────────────────────────────────────────

                    messages.success(request, f'Analysis completed with {ecg_record.confidence:.1f}% confidence')
                    return redirect('ecg_result', ecg_id=ecg_record.id)
                else:
                    ecg_record.status = 'failed'
                    ecg_record.save()
                    messages.error(request, 'Failed to analyze ECG. Please try again.')
                    return redirect('upload')
                
            except Exception as e:
                ecg_record.status = 'failed'
                ecg_record.error_message = str(e)
                ecg_record.save()
                messages.error(request, f'Error processing image: {str(e)}')
                return redirect('upload')
        else:
            # Form is invalid, show errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = ECGUploadForm()
    
    return render(request, 'ecg_app/upload.html', {
        'form': form,
        'recent_ecgs': recent_ecgs
    })

@login_required
def ecg_result_view(request, ecg_id):
    """View ECG analysis result"""
    ecg_record = get_object_or_404(ECGRecord, id=ecg_id, user=request.user)

    probabilities = {
        'Normal ECG': ecg_record.normal_prob,
        'Abnormal Heartbeat': ecg_record.abnormal_prob,
        'Myocardial Infarction': ecg_record.mi_prob,
        'Post MI History': ecg_record.post_mi_prob,
    }

    context = {
        'record': ecg_record,
        'probabilities': probabilities,
    }

    return render(request, 'ecg_app/results.html', context)

@login_required
def ecg_history_view(request):
    """View all ECG records"""
    ecg_records = ECGRecord.objects.filter(user=request.user).order_by('-upload_date')
    
    # Apply filters
    status_filter = request.GET.get('status', 'all')
    if status_filter != 'all':
        ecg_records = ecg_records.filter(status=status_filter)
    
    category_filter = request.GET.get('category', 'all')
    if category_filter != 'all':
        ecg_records = ecg_records.filter(predicted_category=category_filter)
    
    # Date range filter
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    if start_date:
        ecg_records = ecg_records.filter(upload_date__date__gte=start_date)
    if end_date:
        ecg_records = ecg_records.filter(upload_date__date__lte=end_date)
    
    # Calculate statistics
    total_records = ecg_records.count()
    completed_count = ECGRecord.objects.filter(user=request.user, status='completed').count()
    processing_count = ECGRecord.objects.filter(user=request.user, status='processing').count()
    failed_count = ECGRecord.objects.filter(user=request.user, status='failed').count()
    
    # Calculate average confidence
    avg_confidence = ECGRecord.objects.filter(
        user=request.user, 
        status='completed',
        confidence__isnull=False
    ).aggregate(avg_conf=Avg('confidence'))['avg_conf'] or 0
    
    # Get category counts
    category_counts = {}
    category_data = ECGRecord.objects.filter(
        user=request.user, 
        predicted_category__isnull=False
    ).values('predicted_category').annotate(count=Count('predicted_category'))
    
    for item in category_data:
        category_counts[item['predicted_category']] = item['count']
    
    # Get most common category
    most_common = ECGRecord.objects.filter(
        user=request.user, 
        predicted_category__isnull=False
    ).values('predicted_category').annotate(
        count=Count('predicted_category')
    ).order_by('-count').first()
    
    most_common_category = most_common['predicted_category'] if most_common else None
    
    # This month count
    this_month = timezone.now().replace(day=1)
    this_month_count = ECGRecord.objects.filter(
        user=request.user,
        upload_date__gte=this_month
    ).count()
    
    # Pagination
    paginator = Paginator(ecg_records, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_records': total_records,
        'completed_count': completed_count,
        'processing_count': processing_count,
        'failed_count': failed_count,
        'avg_confidence': avg_confidence,
        'categories': ECGRecord.CATEGORY_CHOICES,
        'category_counts': category_counts,
        'most_common_category': most_common_category,
        'this_month_count': this_month_count,
    }
    return render(request, 'ecg_app/history.html', context)

# ========== API VIEWS ==========

@login_required
@csrf_exempt
def api_train_model(request):
    """Simple training endpoint"""
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
                return JsonResponse({
                    'status': 'error',
                    'message': 'Training failed'
                }, status=500)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    return JsonResponse({'error': 'Only POST allowed'}, status=405)

@login_required
def api_user_stats(request):
    """Get user statistics"""
    user_ecgs = ECGRecord.objects.filter(user=request.user)
    total_ecgs = user_ecgs.count()
    
    # Get category distribution
    category_distribution = {}
    category_data = user_ecgs.filter(
        status='completed', 
        predicted_category__isnull=False
    ).values('predicted_category').annotate(count=Count('predicted_category'))
    
    for item in category_data:
        category_distribution[item['predicted_category']] = item['count']
    
    # Calculate normal and abnormal counts
    normal_ecgs = category_distribution.get('normal', 0)
    abnormal_ecgs = total_ecgs - normal_ecgs
    
    # Calculate success rate
    if total_ecgs > 0:
        success_rate = round((normal_ecgs / total_ecgs) * 100, 1)
    else:
        success_rate = 0
    
    return JsonResponse({
        'total_ecgs': total_ecgs,
        'username': request.user.username,
        'category_distribution': category_distribution,
        'normal_ecgs': normal_ecgs,
        'abnormal_ecgs': abnormal_ecgs,
        'success_rate': success_rate,
    })

@login_required
def api_lime_explanation(request, ecg_id):
    """Return LIME explanation image URL and per-class superpixel weights."""
    record = get_object_or_404(ECGRecord, id=ecg_id, user=request.user)
    
    # Check if still processing (if dict doesn't have 4 keys)
    is_ready = bool(record.lime_images and len(record.lime_images) > 0)
    
    return JsonResponse({
        'ecg_id': ecg_id,
        'is_ready': is_ready,
        'lime_images': record.lime_images or {},
        'lime_data': record.lime_explanation_data or {},
        'class_names': ecg_model.class_names,
        'predicted_class': record.predicted_category,
    })


@login_required
def generate_pdf_report(request, ecg_id):
    """
    Generate a PDF report using WeasyPrint containing the original ECG,
    model predictions, and the 4 LIME overlays.
    """
    from django.http import HttpResponse
    from django.template.loader import render_to_string
    import weasyprint

    record = get_object_or_404(ECGRecord, id=ecg_id, user=request.user)
    
    # Check if LIME is done
    lime_ready = bool(record.lime_images and len(record.lime_images) > 0)

    # Build full URLs for WeasyPrint to fetch images
    request_url = request.build_absolute_uri('/')[:-1]
    
    context = {
        'record': record,
        'lime_ready': lime_ready,
        'base_url': request_url,  # needed for local static/media resolving
        'lime_abnormal': f"{request_url}{record.lime_images.get('abnormal', '')}" if lime_ready else None,
        'lime_mi': f"{request_url}{record.lime_images.get('mi', '')}" if lime_ready else None,
        'lime_normal': f"{request_url}{record.lime_images.get('normal', '')}" if lime_ready else None,
        'lime_post_mi': f"{request_url}{record.lime_images.get('post_mi', '')}" if lime_ready else None,
        'original_image_url': f"{request_url}{record.image.url}",
    }

    # Render HTML
    html_string = render_to_string('ecg_app/pdf_report.html', context)
    
    # Generate PDF
    pdf_file = weasyprint.HTML(string=html_string, base_url=request_url).write_pdf()
    
    # Return as attachment
    response = HttpResponse(pdf_file, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="ECG_Report_{record.id}.pdf"'
    
    return response

# ========== ADMIN VIEWS ==========

@staff_member_required
def admin_dashboard_view(request):
    """Admin dashboard - staff only"""
    # Get statistics
    total_users = User.objects.count()
    total_ecgs = ECGRecord.objects.count()
    
    # Active users today
    today = timezone.now().date()
    active_users_today = User.objects.filter(
        Q(last_login__date=today) | Q(date_joined__date=today)
    ).distinct().count()
    
    # Recent users with ECG counts
    # Use the correct related name - check your ECGRecord model
    recent_users = User.objects.annotate(
        ecg_count=Count('ecg_records')  # Changed from 'ecgrecord' to 'ecg_records'
    ).order_by('-date_joined')[:10]
    
    # Recent ECGs
    recent_ecgs = ECGRecord.objects.select_related('user').order_by('-upload_date')[:10]
    
    # Calculate percentages
    active_percentage = int((active_users_today / total_users * 100)) if total_users > 0 else 0
    
    # Mock growth rates
    user_growth = 12
    ecg_growth = 8
    avg_growth = 5
    growth_rate = 15
    
    # ECGs today and new users today
    ecgs_today = ECGRecord.objects.filter(upload_date__date=today).count()
    new_users_today = User.objects.filter(date_joined__date=today).count()
    
    context = {
        'total_users': total_users,
        'total_ecgs': total_ecgs,
        'active_users_today': active_users_today,
        'active_percentage': active_percentage,
        'recent_users': recent_users,
        'recent_ecgs': recent_ecgs,
        'user_growth': user_growth,
        'ecg_growth': ecg_growth,
        'avg_growth': avg_growth,
        'growth_rate': growth_rate,
        'ecgs_today': ecgs_today,
        'new_users_today': new_users_today,
    }
    
    return render(request, 'ecg_app/admin_dashboard.html', context)

def admin_login_view(request):
    """Admin-only login view"""
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(username=username, password=password)
        
        if user is not None and user.is_superuser:
            login(request, user)
            messages.success(request, 'Welcome to Admin Dashboard!')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid admin credentials or insufficient permissions.')
    
    return render(request, 'ecg_app/auth/admin_login.html')

# In views.py, add this function:
@login_required
def export_history_csv_view(request):
    """Export ECG history as CSV"""
    ecg_records = ECGRecord.objects.filter(user=request.user).order_by('-upload_date')
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="ecg_history.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['ID', 'Date', 'Filename', 'Prediction', 'Confidence', 'Status', 'Notes'])
    
    for ecg in ecg_records:
        writer.writerow([
            ecg.id,
            ecg.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            ecg.original_filename,
            ecg.predicted_category,
            f"{ecg.confidence}%" if ecg.confidence else "",
            ecg.status,
            ecg.notes or ""
        ])
    
    return response