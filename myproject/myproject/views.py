from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from .forms import UserUpdateForm


@login_required
def settings_view(request):
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        password_form = PasswordChangeForm(request.user, request.POST)

        if user_form.is_valid():
            user_form.save()
            return redirect('settings')

        if password_form.is_valid():
            user = password_form.save()
            update_session_auth_hash(request, user)  # Keeps the user logged in after password change
            return redirect('settings')

    else:
        user_form = UserUpdateForm(instance=request.user)
        password_form = PasswordChangeForm(request.user)

    return render(request, 'settings.html', {
        'user_form': user_form,
        'password_form': password_form,
    })


def homepage(request):
    # return HttpResponse("Hello World! I'm Home.")
    upload_result = request.session.pop('upload_result', None)  # Get result and clear it from session
    return render(request, 'home.html', {'upload_result': upload_result})


def about(request):
    # return HttpResponse("My About page.")
    return render(request, 'about.html')
