from pathlib import Path
import json
import joblib
import numpy as np

_MODEL = None
_META  = None
_OLS_TABLE = None
_OLS_SUMMARY = None

def get_model():
    global _MODEL
    if _MODEL is None:
        p = Path(__file__).resolve().parent / 'model' / 'model.pkl'
        _MODEL = joblib.load(p)
    return _MODEL

def get_meta():
    global _META
    if _META is None:
        p = Path(__file__).resolve().parent / 'model' / 'model_meta.json'
        _META = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    return _META

def predict_pe(at, v, ap, rh):
    model = get_model()
    X = np.array([[float(at), float(v), float(ap), float(rh)]], dtype=float)
    y = model.predict(X)[0]
    return float(y)

def explain_pe(at, v, ap, rh):
    pipe = get_model()
    scaler = pipe.named_steps.get("scaler")
    reg    = pipe.named_steps.get("model")

    x = np.array([float(at), float(v), float(ap), float(rh)], dtype=float)
    xs = (x - scaler.mean_) / scaler.scale_
    contrib = reg.coef_ * xs
    pred = float(reg.intercept_ + contrib.sum())
    return {
        "prediction_via_linear": pred,
        "intercept": float(reg.intercept_),
        "contributions": {
            "AT": float(contrib[0]),
            "V":  float(contrib[1]),
            "AP": float(contrib[2]),
            "RH": float(contrib[3]),
        }
    }

def get_ols_table():
    """Return dict with meta + table rows from statsmodels OLS."""
    global _OLS_TABLE
    if _OLS_TABLE is None:
        p = Path(__file__).resolve().parent / 'model' / 'ols_table.json'
        _OLS_TABLE = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {"meta": {}, "table": []}
    return _OLS_TABLE

def get_ols_summary():
    """Return full text summary from statsmodels OLS."""
    global _OLS_SUMMARY
    if _OLS_SUMMARY is None:
        p = Path(__file__).resolve().parent / 'model' / 'ols_summary.txt'
        _OLS_SUMMARY = p.read_text(encoding='utf-8') if p.exists() else "No OLS summary found. Train first."
    return _OLS_SUMMARY
