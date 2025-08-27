import io, csv, json, base64
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .forms import PredictForm, InsightsUploadForm
from .utils import predict_pe, explain_pe, get_meta, get_ols_table, get_ols_summary

# ---------- Helpers ----------

def _fig_to_base64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def _colored_scatter_with_fit(x, y, xlabel, ylabel='PE (MW)', title=None):
    """
    Colorful scatter + clearly visible regression line (different color).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,3))
    # Scatter points
    ax.scatter(x, y, s=16, alpha=0.75, label='Data', color='C0')
    # Regression line (contrasting color)
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.min(x), np.max(x), 120)
        ax.plot(xx, m*xx + b, linewidth=2.8, color='C3', label='Trend')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    return _fig_to_base64(fig)

def _contrib_bars(contrib_dict):
    """
    Colorful bar chart for single prediction feature contributions.
    contrib_dict: {"AT": val, "V": val, "AP": val, "RH": val}
    """
    import matplotlib.pyplot as plt
    keys = list(contrib_dict.keys())
    vals = [contrib_dict[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.2,3.2))
    # Bars with distinct colors
    colors = ['C0','C1','C2','C3']
    ax.bar(keys, vals, color=colors[:len(keys)], alpha=0.85)
    ax.axhline(0, color='#666', linewidth=1.0)
    ax.set_title('Feature Contributions (standardized space)')
    ax.set_ylabel('Contribution')
    return _fig_to_base64(fig)

def _csv_to_data_url(content_str, filename="predictions.csv"):
    """
    Create a data: URL to let user download the generated CSV from page.
    Return the href string and the suggested filename.
    """
    b64 = base64.b64encode(content_str.encode('utf-8')).decode('ascii')
    href = f"data:text/csv;base64,{b64}"
    return href, filename

# ---------- Core pages ----------

def index(request):
    """
    Single prediction page.
    Adds a colorful contributions bar chart under the prediction.
    """
    meta = get_meta()
    charts_single = {}
    context = {
        'prediction': None,
        'explanation': None,
        'charts_single': charts_single,
        'form': PredictForm(),
        'meta': meta
    }

    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            at = form.cleaned_data['AT']; v = form.cleaned_data['V']
            ap = form.cleaned_data['AP']; rh = form.cleaned_data['RH']
            pred = round(predict_pe(at, v, ap, rh), 2)
            exp  = explain_pe(at, v, ap, rh)  # has intercept + contributions
            context.update({'prediction': pred, 'explanation': exp, 'form': form})

            # Add a colorful contributions bar chart
            contrib = exp.get('contributions', {})
            if contrib:
                charts_single['contrib_bar'] = _contrib_bars({
                    'AT': contrib.get('AT', 0.0),
                    'V':  contrib.get('V', 0.0),
                    'AP': contrib.get('AP', 0.0),
                    'RH': contrib.get('RH', 0.0),
                })
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
    """
    Batch upload:
    - Input CSV must have AT,V,AP,RH  (no PE column required here).
    - We compute predictions, render colorful charts, and provide a Download button.
    """
    meta = get_meta()
    charts_batch = {}
    download_href = None
    download_name = None
    msg = None

    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        try:
            text = f.read().decode('utf-8-sig', errors='ignore').splitlines()
            reader = csv.DictReader(text)
            rows = list(reader)
            if not rows:
                msg = "CSV appears empty."
            else:
                import pandas as pd
                df = pd.DataFrame(rows)

                # Normalize headers: lowercase->alias->uppercase keys
                alias = {
                    "at":"AT","ambient temperature":"AT","ambient_temperature":"AT",
                    "v":"V","exhaust vacuum":"V","vacuum":"V","exhaust_vacuum":"V",
                    "ap":"AP","ambient pressure":"AP","ambient_pressure":"AP",
                    "atm pressure":"AP","atm_pressure":"AP",
                    "rh":"RH","relative humidity":"RH","relative_humidity":"RH",
                }
                newcols = {}
                for c in df.columns:
                    key = c.lower().strip()
                    if key in alias:
                        newcols[c] = alias[key]
                    elif key in {"at","v","ap","rh"}:
                        newcols[c] = key.upper()
                if newcols:
                    df = df.rename(columns=newcols)

                needed = ['AT','V','AP','RH']
                miss = [c for c in needed if c not in df.columns]
                if miss:
                    msg = f"Missing columns: {', '.join(miss)}"
                else:
                    # cast + drop NaNs
                    for c in needed:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    df = df.dropna(subset=needed)
                    if df.empty:
                        msg = "All rows became NaN after parsing numbers. Check your CSV values."
                    else:
                        # Predict
                        import numpy as np
                        X = df[['AT','V','AP','RH']].to_numpy(float)
                        preds = []
                        for row in X:
                            preds.append(predict_pe(row[0], row[1], row[2], row[3]))
                        df['PE_pred'] = np.round(preds, 2)

                        # Build downloadable CSV
                        out = io.StringIO()
                        fieldnames = ['AT','V','AP','RH','PE_pred']
                        w = csv.DictWriter(out, fieldnames=fieldnames)
                        w.writeheader()
                        for _, r in df.iterrows():
                            w.writerow({
                                'AT': r['AT'], 'V': r['V'], 'AP': r['AP'], 'RH': r['RH'],
                                'PE_pred': f"{r['PE_pred']:.2f}"
                            })
                        csv_content = out.getvalue()
                        download_href, download_name = _csv_to_data_url(csv_content)

                        # Colorful charts (use PE_pred)
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        # 1) Histogram of PE_pred
                        fig1, ax1 = plt.subplots(figsize=(5,3))
                        ax1.hist(df['PE_pred'].values, bins=30, color='C2', alpha=0.85)
                        ax1.set_title('Predicted PE Distribution')
                        ax1.set_xlabel('PE_pred (MW)'); ax1.set_ylabel('Count')
                        charts_batch['pe_hist'] = _fig_to_base64(fig1)

                        # 2) Scatters with colorful regression lines
                        charts_batch['pe_vs_at'] = _colored_scatter_with_fit(
                            df['AT'].values, df['PE_pred'].values, 'AT (°C)', 'PE_pred (MW)', 'PE_pred vs AT'
                        )
                        charts_batch['pe_vs_v'] = _colored_scatter_with_fit(
                            df['V'].values, df['PE_pred'].values, 'V (cm Hg)', 'PE_pred (MW)', 'PE_pred vs V'
                        )
                        charts_batch['pe_vs_ap'] = _colored_scatter_with_fit(
                            df['AP'].values, df['PE_pred'].values, 'AP (mbar)', 'PE_pred (MW)', 'PE_pred vs AP'
                        )
                        charts_batch['pe_vs_rh'] = _colored_scatter_with_fit(
                            df['RH'].values, df['PE_pred'].values, 'RH (%)', 'PE_pred (MW)', 'PE_pred vs RH'
                        )
        except Exception as e:
            msg = f"Failed to parse CSV: {e}"

        # Render the page showing charts + download link instead of direct file return
        return render(request, 'predictor/batch.html', {
            "meta": meta,
            "charts_batch": charts_batch,
            "download_href": download_href,
            "download_name": download_name,
            "msg": msg
        })

    # GET or no file -> initial page
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
    Upload a CSV (AT,V,AP, RH, PE) to view summary stats, correlations, and colorful charts.
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
                    import numpy as np
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
                        elif key in {"at","v","ap","rh","pe"}:
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

                            # Colorful charts
                            import matplotlib
                            matplotlib.use("Agg")
                            import matplotlib.pyplot as plt

                            # 1) Histogram of PE
                            fig1, ax1 = plt.subplots(figsize=(5,3))
                            ax1.hist(df['PE'].values, bins=30, color='C2', alpha=0.85)
                            ax1.set_title('PE Distribution')
                            ax1.set_xlabel('PE (MW)'); ax1.set_ylabel('Count')
                            charts['pe_hist'] = _fig_to_base64(fig1)

                            # 2) Scatters with colorful regression lines
                            charts['pe_vs_at'] = _colored_scatter_with_fit(
                                df['AT'].values, df['PE'].values, 'AT (°C)', 'PE (MW)', 'PE vs AT'
                            )
                            charts['pe_vs_v'] = _colored_scatter_with_fit(
                                df['V'].values, df['PE'].values, 'V (cm Hg)', 'PE (MW)', 'PE vs V'
                            )
                            charts['pe_vs_ap'] = _colored_scatter_with_fit(
                                df['AP'].values, df['PE'].values, 'AP (mbar)', 'PE (MW)', 'PE vs AP'
                            )
                            charts['pe_vs_rh'] = _colored_scatter_with_fit(
                                df['RH'].values, df['PE'].values, 'RH (%)', 'PE (MW)', 'PE vs RH'
                            )

                            # Optional: correlation heatmap
                            try:
                                figH, axH = plt.subplots(figsize=(4.2,3.8))
                                im = axH.imshow(corr.values, aspect='auto', cmap='viridis')
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


def ols(request):
    """Show OLS coefficient table and full statsmodels summary."""
    meta = get_meta()
    ols_tbl = get_ols_table()
    ols_sum = get_ols_summary()
    return render(request, "predictor/ols.html", {
        "meta": meta,
        "ols": ols_tbl,
        "summary": ols_sum
    })