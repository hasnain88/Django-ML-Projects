from pathlib import Path
import json
import joblib
import numpy as np

_MODEL = None
_META  = None

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
        if p.exists():
            _META = json.loads(p.read_text(encoding='utf-8'))
        else:
            _META = {}
    return _META

def predict_pe(at, v, ap, rh):
    model = get_model()
    X = np.array([[float(at), float(v), float(ap), float(rh)]], dtype=float)
    y = model.predict(X)[0]
    return float(y)

def explain_pe(at, v, ap, rh):
    """
    Returns per-feature contribution using the inner LinearRegression on scaled inputs:
    contribution_i = coef_i * x_scaled_i ;  sum(contrib) + intercept_ = prediction (in scaled space).
    This is a simple, transparent explanation for linear models with a scaler.
    """
    pipe = get_model()
    scaler = pipe.named_steps.get("scaler")
    reg    = pipe.named_steps.get("model")

    x = np.array([float(at), float(v), float(ap), float(rh)], dtype=float)
    xs = (x - scaler.mean_) / scaler.scale_           # scaled
    contrib = reg.coef_ * xs                          # per-feature terms
    pred = float(reg.intercept_ + contrib.sum())      # equals pipe.predict(x) numerically
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
