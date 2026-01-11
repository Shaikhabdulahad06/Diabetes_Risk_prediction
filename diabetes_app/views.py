from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
import re
import numpy as np
import pickle
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'ml_model/risk_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Signup view
def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm = request.POST.get('confirm', '')
        email = request.POST.get('email', '').strip()  # optional

        errors = []

        # Basic presence checks
        if not username or not password or not confirm:
            errors.append("All fields are required.")

        # Username rules
        if username and (len(username) < 3 or len(username) > 150):
            errors.append("Username must be between 3 and 150 characters.")
        if username and not re.match(r'^[A-Za-z0-9_]+$', username):
            errors.append("Username may contain only letters, numbers and underscores.")

        # Password checks
        if password != confirm:
            errors.append("Passwords do not match.")
        if password and len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit.")

        # Uniqueness
        if username and User.objects.filter(username=username).exists():
            errors.append("Username already exists.")
        if email and User.objects.filter(email=email).exists():
            errors.append("Email already registered.")

        # Show errors or create user
        if errors:
            for e in errors:
                messages.error(request, e)
            return redirect('signup')

        try:
            User.objects.create_user(username=username, email=email or None, password=password)
            messages.success(request, "Account created successfully. Please login.")
            return redirect('login')
        except IntegrityError:
            messages.error(request, "A database error occurred. Please try again.")
            return redirect('signup')

    return render(request, 'signup.html')


# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, "Username and password are required.")
            return redirect('login')

        # Simple session-based brute-force protection
        failed = request.session.get('failed_logins', 0)
        if failed >= 5:
            messages.error(request, "Too many failed attempts. Try again later.")
            return redirect('login')

        user = authenticate(request, username=username, password=password)
        if user is None:
            request.session['failed_logins'] = failed + 1
            attempts_left = max(0, 5 - request.session['failed_logins'])
            messages.error(request, f"Invalid credentials. Attempts left: {attempts_left}")
            return redirect('login')

        if not user.is_active:
            messages.error(request, "Account is inactive. Contact the administrator.")
            return redirect('login')

        login(request, user)
        request.session['failed_logins'] = 0
        messages.success(request, "Logged in successfully.")
        return redirect('index')

    return render(request, 'login.html')


# Logout view
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


# Homepage / Index
def index(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'index.html')


# Diabetes risk prediction
def predict(request):
    if not request.user.is_authenticated:
        return redirect('login')

    if request.method == 'POST':
        try:
            input_data = [
                float(request.POST['pregnancies']),
                float(request.POST['bmi']),
                float(request.POST['dpf']),
                float(request.POST['age']),
            ]
        except (ValueError, KeyError):
            messages.error(request, "Invalid input values. Please check all fields.")
            return redirect('index')

        try:
            prob = model.predict_proba([input_data])[0][1]
        except Exception:
            messages.error(request, "Prediction failed due to a model error. Contact admin.")
            return redirect('index')

        # Map probability → risk category
        if prob < 0.33:
            risk_level = "Low"
            advice = [
                "Maintain a balanced diet",
                "Exercise at least 30 minutes daily",
                "Go for a health checkup once a year"
            ]
        elif prob < 0.66:
            risk_level = "Medium"
            advice = [
                "Monitor your BMI regularly",
                "Adopt a low-sugar, high-fiber diet",
                "Check blood glucose occasionally",
                "Consult a doctor if symptoms appear"
            ]
        else:
            risk_level = "High"
            advice = [
                "Consult a doctor immediately",
                "Regularly monitor glucose and blood pressure",
                "Follow a strict diet and exercise plan",
                "Discuss possible medications with your doctor"
            ]

        return render(request, 'result.html', {
            'risk_level': risk_level,
            'probability': round(prob, 2),
            'advice': advice
        })

    return render(request, 'index.html')

def bmi_calculator(request):
    if request.method == 'POST':
        try:
            feet = float(request.POST.get('height_feet', 0))
            inches = float(request.POST.get('height_inches', 0))
            weight = float(request.POST.get('weight', 0))

            # Convert height to meters
            total_inches = feet * 12 + inches
            height_m = total_inches * 0.0254

            if height_m <= 0 or weight <= 0:
                messages.error(request, "Please enter valid positive numbers.")
                return redirect('bmi')

            bmi = weight / (height_m ** 2)
            return render(request, 'bmi_result.html', {'bmi': round(bmi, 2)})

        except ValueError:
            messages.error(request, "Invalid input values. Please check again.")
            return redirect('bmi')

    return render(request, 'bmi.html')

