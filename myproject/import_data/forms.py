from django import forms
from .models import EEGFile

class EEGFileUploadForm(forms.ModelForm):
    class Meta:
        model = EEGFile
        fields = ['file']
