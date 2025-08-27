import io, csv, json
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .forms import PredictForm
from .utils import predict_pe, explain_pe, get_meta

def index(request):
    meta = get_meta()
    context = {'prediction': None, 'explanation': None, 'form': PredictForm(), 'meta': meta}
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            at = form.cleaned_data['AT']; v = form.cleaned_data['V']
            ap = form.cleaned_data['AP']; rh = form.cleaned_data['RH']
            pred = round(predict_pe(at, v, ap, rh), 2)
            exp  = explain_pe(at, v, ap, rh)
            # NOTE: exp.prediction_via_linear equals pipeline prediction up to tiny float tolerance
            context.update({'prediction': pred, 'explanation': exp, 'form': form})
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
    pred = round(predict_pe(at, v, ap, rh), 2)
    return JsonResponse({'PE': pred})

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

    meta = get_meta()
    return render(request, 'predictor/batch.html', {"meta": meta})

def sample_csv(request):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["AT","V","AP","RH"])
    w.writerow([14.96,41.76,1024.07,73.17])
    w.writerow([25.18,62.96,1020.04,59.08])
    w.writerow([5.11,39.40,1012.16,92.14])
    resp = HttpResponse(out.getvalue(), content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="sample_input.csv"'
    return resp

def about(request):
    meta = get_meta()
    return render(request, "predictor/about.html", {"meta": meta})
