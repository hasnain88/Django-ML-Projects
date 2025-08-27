import io, csv, json, base64
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .forms import PredictForm, InsightsUploadForm
from .utils import predict_pe, explain_pe, get_meta

# ---------- Helpers ----------

def _fig_to_base64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

# ---------- Core pages ----------

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
        decoded = f.read().decode('utf-8', errors='ignore').splitlines()
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

# ---------- Insights (tables + charts) ----------

def insights(request):
    """
    Upload a CSV (AT,V,AP, RH, PE) to view summary stats, correlations, and charts.
    Header normalization handles lowercase, spaces, BOM, and common aliases.
    """
    meta = get_meta()
    charts = {}
    tables = {}
    msg = None
    form = InsightsUploadForm()

    if request.method == 'POST' and request.FILES.get('file'):
        form = InsightsUploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES['file']
            try:
                # utf-8-sig handles BOM; DictReader manages commas safely
                text = f.read().decode('utf-8-sig', errors='ignore').splitlines()
                reader = csv.DictReader(text)
                rows = list(reader)
                if not rows:
                    msg = "CSV appears empty."
                else:
                    import pandas as pd
                    df = pd.DataFrame(rows)

                    # --- normalize headers ---
                    alias = {
                        "at":"AT","ambient temperature":"AT","ambient_temperature":"AT",
                        "v":"V","exhaust vacuum":"V","vacuum":"V","exhaust_vacuum":"V",
                        "ap":"AP","ambient pressure":"AP","ambient_pressure":"AP",
                        "atm pressure":"AP","atm_pressure":"AP",
                        "rh":"RH","relative humidity":"RH","relative_humidity":"RH",
                        "pe":"PE","power output":"PE","energy output":"PE",
                        "net hourly electrical energy output":"PE",
                        "net energy":"PE","net energy output":"PE",
                    }
                    newcols = {}
                    for c in df.columns:
                        key = c.lower().strip()
                        if key in alias:
                            newcols[c] = alias[key]
                        else:
                            # also standardize exact keys to uppercase if already the same name in different case
                            if key in {"at","v","ap","rh","pe"}:
                                newcols[c] = key.upper()
                    if newcols:
                        df = df.rename(columns=newcols)

                    needed = ['AT','V','AP','RH','PE']
                    miss = [c for c in needed if c not in df.columns]
                    if miss:
                        msg = f"Missing columns: {', '.join(miss)}"
                    else:
                        # cast to numeric and drop NaNs
                        for c in needed:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                        df = df.dropna(subset=needed)
                        if df.empty:
                            msg = "All rows became NaN after parsing numbers. Check your CSV values."
                        else:
                            # Summary table
                            desc = df[needed].describe().round(3)
                            tables['summary_html'] = desc.to_html(classes="table table-sm table-striped")

                            # Correlation table
                            corr = df[needed].corr(numeric_only=True).round(3)
                            tables['corr_html'] = corr.to_html(classes="table table-sm table-striped")

                            # Charts
                            import matplotlib
                            matplotlib.use("Agg")
                            import matplotlib.pyplot as plt
                            import numpy as np

                            # 1) PE Histogram
                            fig1, ax1 = plt.subplots(figsize=(5,3))
                            ax1.hist(df['PE'].values, bins=30)
                            ax1.set_title('PE Distribution')
                            ax1.set_xlabel('PE (MW)'); ax1.set_ylabel('Count')
                            charts['pe_hist'] = _fig_to_base64(fig1)

                            # Scatter with linear fit helper
                            def scatter_with_fit(x, y, xlabel):
                                fig, ax = plt.subplots(figsize=(5,3))
                                ax.scatter(x, y, s=8, alpha=0.7)
                                if len(x) > 1:
                                    m, b = np.polyfit(x, y, 1)
                                    xx = np.linspace(np.min(x), np.max(x), 100)
                                    ax.plot(xx, m*xx + b)
                                ax.set_xlabel(xlabel); ax.set_ylabel('PE (MW)')
                                ax.set_title(f'PE vs {xlabel}')
                                return _fig_to_base64(fig)

                            charts['pe_vs_at'] = scatter_with_fit(df['AT'].values, df['PE'].values, 'AT (°C)')
                            charts['pe_vs_v']  = scatter_with_fit(df['V'].values,  df['PE'].values, 'V (cm Hg)')
                            charts['pe_vs_ap'] = scatter_with_fit(df['AP'].values, df['PE'].values, 'AP (mbar)')
                            charts['pe_vs_rh'] = scatter_with_fit(df['RH'].values, df['PE'].values, 'RH (%)')

                            # Optional: correlation heatmap
                            try:
                                figH, axH = plt.subplots(figsize=(4.2,3.8))
                                im = axH.imshow(corr.values, aspect='auto')
                                axH.set_xticks(range(len(needed))); axH.set_xticklabels(needed, rotation=45, ha='right')
                                axH.set_yticks(range(len(needed))); axH.set_yticklabels(needed)
                                axH.set_title('Correlation Heatmap')
                                figH.colorbar(im, ax=axH, fraction=0.046)
                                charts['heatmap'] = _fig_to_base64(figH)
                            except Exception:
                                pass

            except Exception as e:
                msg = f"Failed to parse CSV: {e}"

    return render(request, "predictor/insights.html", {
        "form": form, "meta": meta, "charts": charts, "tables": tables, "msg": msg
    })
