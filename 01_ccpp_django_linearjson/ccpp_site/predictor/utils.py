import json
from pathlib import Path

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        p = Path(__file__).resolve().parent / 'model' / 'model.json'
        with open(p, 'r', encoding='utf-8') as f:
            _MODEL = json.load(f)
    return _MODEL

def predict_pe(at, v, ap, rh):
    m = get_model()
    coef = m['coef']          # [AT, V, AP, RH]
    intercept = m['intercept']
    return intercept + coef[0]*at + coef[1]*v + coef[2]*ap + coef[3]*rh
