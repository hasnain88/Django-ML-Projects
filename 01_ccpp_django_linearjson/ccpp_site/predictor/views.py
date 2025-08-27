import io, csv, json
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .forms import PredictForm
from .utils import predict_pe

def index(request):
    context = {'prediction': None, 'form': PredictForm()}
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            at = form.cleaned_data['AT']; v = form.cleaned_data['V']
            ap = form.cleaned_data['AP']; rh = form.cleaned_data['RH']
            pred = float(predict_pe(at, v, ap, rh))
            context.update({'prediction': round(pred, 2), 'form': form})
        else:
            context['form'] = form
    return render(request, 'predictor/index.html', context)

def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST JSON with fields AT, V, AP, RH'}, status=405)
    try:
        payload = json.loads(request.body.decode('utf-8'))
        at = float(payload['AT']); v = float(payload['V'])
        ap = float(payload['AP']); rh = float(payload['RH'])
    except Exception as e:
        return JsonResponse({'error': f'Invalid JSON: {e}'}, status=400)
    pred = float(predict_pe(at, v, ap, rh))
    return JsonResponse({'PE': round(pred, 2)})

def batch_predict(request):
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        decoded = f.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded)
        rows = list(reader)
        out = io.StringIO()
        fieldnames = ['AT','V','AP','RH','PE_pred']
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            at = float(r['AT']); v = float(r['V']); ap = float(r['AP']); rh = float(r['RH'])
            pe = predict_pe(at, v, ap, rh)
            w.writerow({'AT': r['AT'], 'V': r['V'], 'AP': r['AP'], 'RH': r['RH'], 'PE_pred': f"{pe:.2f}"})
        resp = HttpResponse(out.getvalue(), content_type='text/csv')
        resp['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        return resp
    return render(request, 'predictor/batch.html')
