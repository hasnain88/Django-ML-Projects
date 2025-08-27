from django import forms

class PredictForm(forms.Form):
    AT = forms.FloatField(label='Ambient Temperature (AT, °C)')
    V  = forms.FloatField(label='Exhaust Vacuum (V, cm Hg)')
    AP = forms.FloatField(label='Ambient Pressure (AP, mbar)')
    RH = forms.FloatField(label='Relative Humidity (RH, %)')
