from django import forms

class PredictForm(forms.Form):
    AT = forms.FloatField(label='Ambient Temperature (AT, °C)', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    V  = forms.FloatField(label='Exhaust Vacuum (V, cm Hg)', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    AP = forms.FloatField(label='Ambient Pressure (AP, mbar)', widget=forms.NumberInput(attrs={'class': 'form-control'}))
    RH = forms.FloatField(label='Relative Humidity (RH, %)', widget=forms.NumberInput(attrs={'class': 'form-control'}))

class InsightsUploadForm(forms.Form):
    file = forms.FileField(
        label="Upload CSV (AT,V,AP,RH,PE)",
        widget=forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )