from django.shortcuts import render, redirect
from .forms import EEGFileUploadForm
from .models import EEGFile
from .processing import preprocess_eeg, run_ml_model  # Assuming these are processing functions

def import_data_view(request):
    if request.method == 'POST':
        form = EEGFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            eeg_file = form.save()  # Save the uploaded file
            # Placeholder: process the file and get the result
            processed_data = preprocess_eeg(eeg_file.file.path)
            ml_result = run_ml_model(processed_data)
            
            # Option 1: Pass result directly to the template
            return render(request, 'import_data/result.html', {'result': ml_result})
            
            # Option 2: Redirect to result page (if you prefer a separate result view)
            # request.session['ml_result'] = ml_result
            # return redirect('import_data_result')

    else:
        form = EEGFileUploadForm()

    return render(request, 'import_data/upload.html', {'form': form})

