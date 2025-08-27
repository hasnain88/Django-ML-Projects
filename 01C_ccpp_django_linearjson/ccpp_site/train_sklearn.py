import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib, json, time, statsmodels.api as sm

EXCEL_PATH = r"01_Combined Cycle Power Plant.xlsx"  # update path if needed

# --- Load & normalize columns ---
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
df.columns = [c.strip() for c in df.columns]
aliases = {
    "at":"AT","ambient temperature":"AT","ambient_temperature":"AT",
    "v":"V","exhaust vacuum":"V","vacuum":"V","exhaust_vacuum":"V",
    "ap":"AP","ambient pressure":"AP","ambient_pressure":"AP","atm pressure":"AP","atm_pressure":"AP",
    "rh":"RH","relative humidity":"RH","relative_humidity":"RH",
    "pe":"PE","power output":"PE","energy output":"PE","net hourly electrical energy output":"PE",
    "net energy":"PE","net energy output":"PE",
}
df = df.rename(columns={c: aliases.get(c.lower().strip(), c) for c in df.columns})
df = df[["AT","V","AP","RH","PE"]].dropna().copy()

# --- Data & split ---
X = df[["AT","V","AP","RH"]].to_numpy(dtype=float)
y = df["PE"].to_numpy(dtype=float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Pipeline for prediction ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe.fit(Xtr, ytr)

# --- Metrics (pipeline) ---
y_pred = pipe.predict(Xte)
rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
r2   = float(r2_score(yte, y_pred))
print(f"[Pipeline] R²: {r2:.4f} | RMSE: {rmse:.3f} | Train: {len(Xtr)} | Test: {len(Xte)}")

# --- Save pipeline model ---
out_dir = Path(__file__).resolve().parent / "predictor" / "model"
out_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, out_dir / "model.pkl")
print("Saved pipeline:", out_dir / "model.pkl")

# --- OLS on ORIGINAL (unscaled) features ---
X_ols = sm.add_constant(X)          # adds intercept term
ols_model = sm.OLS(y, X_ols).fit()
ols_summary_text = ols_model.summary().as_text()

# Build a compact JSON table for the UI
params = ols_model.params
bse    = ols_model.bse
tvals  = ols_model.tvalues
pvals  = ols_model.pvalues
conf   = ols_model.conf_int(alpha=0.05)
index  = ["Intercept","AT","V","AP","RH"]
table = []
for i, name in enumerate(index):
    table.append({
        "term": name,
        "coef": float(params[i]),
        "std_err": float(bse[i]),
        "t": float(tvals[i]),
        "p": float(pvals[i]),
        "ci_low": float(conf[i,0]),
        "ci_high": float(conf[i,1])
    })

ols_meta = {
    "nobs": int(ols_model.nobs),
    "df_model": float(ols_model.df_model),
    "df_resid": float(ols_model.df_resid),
    "r2": float(ols_model.rsquared),
    "r2_adj": float(ols_model.rsquared_adj),
    "fvalue": float(ols_model.fvalue) if ols_model.fvalue is not None else None,
    "f_pvalue": float(ols_model.f_pvalue) if ols_model.f_pvalue is not None else None,
    "aic": float(ols_model.aic),
    "bic": float(ols_model.bic),
}

# Save OLS artifacts
(out_dir / "ols_summary.txt").write_text(ols_summary_text, encoding="utf-8")
(out_dir / "ols_table.json").write_text(json.dumps({"meta": ols_meta, "table": table}, indent=2), encoding="utf-8")
print("Saved OLS:", out_dir / "ols_summary.txt", "and", out_dir / "ols_table.json")

# --- Save pipeline metadata for UI ---
meta = {
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "algorithm": "StandardScaler + LinearRegression",
    "target": "PE (MW)",
    "features": [
        {"name": "AT", "label": "Ambient Temperature (°C)"},
        {"name": "V",  "label": "Exhaust Vacuum (cm Hg)"},
        {"name": "AP", "label": "Ambient Pressure (mbar)"},
        {"name": "RH", "label": "Relative Humidity (%)"},
    ],
    "ranges": {
        "AT": {"min": float(df["AT"].min()), "max": float(df["AT"].max())},
        "V":  {"min": float(df["V"].min()),  "max": float(df["V"].max())},
        "AP": {"min": float(df["AP"].min()), "max": float(df["AP"].max())},
        "RH": {"min": float(df["RH"].min()), "max": float(df["RH"].max())},
    },
    "metrics": {"r2": r2, "rmse": rmse, "n_train": len(Xtr), "n_test": len(Xte), "n_total": len(X)},
}
(out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Saved pipeline meta:", out_dir / "model_meta.json")
