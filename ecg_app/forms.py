# forms.py - CORRECTED with 'image' field
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import ECGRecord, Patient, UserProfile


def validate_email_domain(email):
    """
    Checks that the email's domain has real, working MX (mail) records.
    - NXDOMAIN           -> domain doesn't exist        -> reject
    - NoAnswer           -> no MX record at all          -> reject
    - MX -> localhost/.  -> broken/placeholder MX        -> reject
    - Timeout/network    -> allow (don't block on blip)
    """
    if not email:
        return
    domain = email.split('@')[-1]
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, 'MX', lifetime=5)

        # Reject placeholder MX records (localhost / null MX '.')
        invalid_targets = {'localhost', 'localhost.', '.'}
        all_invalid = all(
            str(r.exchange).rstrip('.').lower() in invalid_targets
            for r in answers
        )
        if all_invalid:
            raise forms.ValidationError(
                f'The domain "{domain}" does not have a working mail server. '
                f'Please enter a valid email address (e.g. name@gmail.com).'
            )

    except dns.resolver.NXDOMAIN:
        raise forms.ValidationError(
            f'The domain "{domain}" does not exist. '
            f'Please enter a valid email address (e.g. name@gmail.com).'
        )
    except dns.resolver.NoAnswer:
        raise forms.ValidationError(
            f'The domain "{domain}" does not appear to accept email. '
            f'Please enter a valid email address.'
        )
    except forms.ValidationError:
        raise   # re-raise our own error
    except Exception:
        # Network timeout or DNS unavailable — accept the address
        # so a connectivity blip never blocks a legitimate user.
        pass


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email'
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your first name'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your last name'
        })
    )
    
    role = forms.ChoiceField(
        choices=UserProfile.ROLE_CHOICES,
        required=True,
        initial='user',
        label='Account Type',
        widget=forms.Select(attrs={
            'class': 'form-select mb-3'
        })
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
        
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Choose a username'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm password'})

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("A user with that username already exists.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("A user with that email already exists.")
        validate_email_domain(email)
        return email


class UserLoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter username'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter password'
        })
    )

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    first_name = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
        }

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['name', 'email', 'age', 'gender', 'contact_number', 'medical_history']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email Address (Optional)'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Age'}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'contact_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Phone Number (Optional)'}),
            'medical_history': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Prior conditions, medications (Optional)...'}),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip()
        if not email:
            return email   # field is optional — skip validation
        validate_email_domain(email)
        return email

class ECGUploadForm(forms.ModelForm):
    class Meta:
        model = ECGRecord
        fields = ['patient', 'image', 'notes']
        widgets = {
            'patient': forms.Select(attrs={
                'class': 'form-control',
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*,.pdf',
                'required': True
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional notes about this ECG...'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        # We need to filter patients by the current user
        # We will pass the user to the form init in views.py
        user = kwargs.pop('user', None)
        super(ECGUploadForm, self).__init__(*args, **kwargs)
        
        # Make patient optional
        self.fields['patient'].required = False
        self.fields['patient'].empty_label = "--- Select Patient (Optional) ---"
        
        if user:
            if hasattr(user, 'profile') and user.profile.role == 'clinic':
                self.fields['patient'].queryset = Patient.objects.filter(user=user)
            else:
                self.fields['patient'].widget = forms.HiddenInput()

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (10MB limit)
            max_size = 10 * 1024 * 1024  # 10MB
            if image.size > max_size:
                raise forms.ValidationError(f'File size must be under {max_size/1024/1024}MB')
            
            # Check file extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']
            import os
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(f'Unsupported file format. Supported formats: {", ".join(valid_extensions)}')
        
        return image