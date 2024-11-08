from django.shortcuts import render, redirect
from .models import EEGFile
from .forms import EEGFileUploadForm
from .processing import preprocess_eeg, run_ml_model
import os
import mne
import matplotlib
matplotlib.use('Agg')
from django.conf import settings

def import_data_view(request):
    if request.method == 'POST':
        form = EEGFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            eeg_file = form.save()  # Save the uploaded file
            file_path = os.path.join(settings.MEDIA_ROOT, eeg_file.file.name)  # Full path to the uploaded file
            
            # Run the preprocessing function
            processed_data, PSD_data = preprocess_eeg(file_path)
            
            # Run the ML model on the processed data
            result = run_ml_model(processed_data, PSD_data)
            
            # Save the EEG plot as an image file
            eeg_plot_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'eeg_plot.png')
            EEG_image = processed_data.plot(scalings='auto', show=False, block=False)
            EEG_image.set_size_inches(40,10)
            EEG_image.savefig(eeg_plot_path, dpi=300, bbox_inches='tight')  # Save the matplotlib figure as an image
            #EEG_image.close()  # Close the plot to free memory
            
            # Create a URL for the plot
            eeg_plot_url = os.path.join(settings.MEDIA_URL, 'uploads', 'eeg_plot.png')
            
            # Pass the result and plot URL to the template
            context = {
                'result': result,
                'eeg_plot_url': eeg_plot_url  # URL to access the EEG plot image
            }
            return render(request, 'import_data/result.html', context)
    else:
        form = EEGFileUploadForm()
    
    return render(request, 'import_data/upload.html', {'form': form})
