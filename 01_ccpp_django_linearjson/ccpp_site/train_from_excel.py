import json
from pathlib import Path
import numpy as np
import pandas as pd

# === CONFIG ===
EXCEL_PATH = r"01_Combined Cycle Power Plant.xlsx"  # put your file here (same folder as manage.py or use absolute path)

# Load Excel
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
df.columns = [c.strip() for c in df.columns]

# Normalize column names
aliases = {
    "at": "AT", "ambient temperature": "AT", "ambient_temperature": "AT",
    "v": "V", "exhaust vacuum": "V", "vacuum": "V", "exhaust_vacuum": "V",
    "ap": "AP", "ambient pressure": "AP", "ambient_pressure": "AP", "atm pressure": "AP", "atm_pressure": "AP",
    "rh": "RH", "relative humidity": "RH", "relative_humidity": "RH",
    "pe": "PE", "power output": "PE", "energy output": "PE", "net hourly electrical energy output": "PE",
    "net energy": "PE", "net energy output": "PE",
}
rename_map = {}
for c in df.columns:
    key = c.lower().strip()
    if key in aliases: rename_map[c] = aliases[key]
if rename_map:
    df = df.rename(columns=rename_map)

df = df[["AT","V","AP","RH","PE"]].dropna().copy()

# Prepare arrays
X = df[["AT","V","AP","RH"]].to_numpy(dtype=float)
y = df["PE"].to_numpy(dtype=float)

# Train/test split for metrics (80/20)
rng = np.random.RandomState(42)
perm = rng.permutation(len(X))
split = int(0.8 * len(X))
idx_tr, idx_te = perm[:split], perm[split:]
Xtr, Xte = X[idx_tr], X[idx_te]
ytr, yte = y[idx_tr], y[idx_te]

# Fit linear regression via normal equation (no sklearn dependency)
Xb = np.hstack([np.ones((Xtr.shape[0],1)), Xtr])  # add bias
theta, *_ = np.linalg.lstsq(Xb, ytr, rcond=None)
intercept = float(theta[0])
coef = theta[1:].tolist()

# Evaluate
y_pred = intercept + Xte @ np.array(coef)
ss_res = float(np.sum((yte - y_pred)**2))
ss_tot = float(np.sum((yte - np.mean(yte))**2))
r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
rmse = float(np.sqrt(np.mean((yte - y_pred)**2)))

# Save model.json
model = {
    "runtime": "numpy-lstsq",
    "intercept": intercept,
    "coef": coef,  # order: AT, V, AP, RH
    "feature_order": ["AT","V","AP","RH"],
    "metrics": {"r2": r2, "rmse": rmse, "n": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
}

out_path = Path(__file__).resolve().parent / "predictor" / "model" / "model.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)

print("Saved:", out_path)
print("Metrics -> R2:", model["metrics"]["r2"], " RMSE:", model["metrics"]["rmse"])
