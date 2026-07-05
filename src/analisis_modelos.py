from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CONSENT_VALUE = "Sí, acepto participar."


def cronbach_alpha(frame: pd.DataFrame) -> float:
    data = frame.dropna()
    k = data.shape[1]
    if k < 2:
        return float("nan")
    item_var = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def recode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df[df.columns[1]].eq(CONSENT_VALUE)].copy()
    cols = {i: df.columns[i] for i in range(len(df.columns))}

    age_map = {"17–18": 17.5, "19–20": 19.5, "21–22": 21.5, "23–24": 23.5, "25 o más": 25.0}
    year_map = {"Primer año": 1, "Segundo año": 2, "Tercer año": 3, "Cuarto año": 4, "Quinto año o más": 5}
    work_map = {"No": 0, "Sí, menos de 20 horas semanales": 1, "Sí, 20 horas semanales o más": 1}
    free_map = {"Menos de 1 hora": 0.5, "1–2 horas": 1.5, "3–4 horas": 3.5, "5 horas o más": 5.5}
    gpa_map = {"Menos de 1.00": 0.75, "1.00–1.49": 1.25, "1.50–1.99": 1.75, "2.00–2.49": 2.25, "2.50–3.00": 2.75}
    perf_map = {"Muy bajo": 1, "Bajo": 2, "Regular": 3, "Bueno": 4, "Muy bueno": 5}
    sleep_hours_map = {"Menos de 4 horas": 3.5, "4–5 horas": 4.5, "6–7 horas": 6.5, "8–9 horas": 8.5, "Más de 9 horas": 9.5}
    sleep_adequacy_map = {"Menos de 4 horas": 1, "4–5 horas": 2, "6–7 horas": 4, "8–9 horas": 5, "Más de 9 horas": 4}

    df["edad_num"] = df[cols[2]].map(age_map)
    df["sexo"] = df[cols[3]]
    df["sex_male"] = np.where(df["sexo"].eq("Masculino"), 1.0, np.where(df["sexo"].eq("Femenino"), 0.0, np.nan))
    df["facultad"] = df[cols[4]]
    df["year_num"] = df[cols[5]].map(year_map)
    df["trabaja"] = df[cols[6]].map(work_map)
    df["free_hours"] = df[cols[7]].map(free_map)
    df["gpa"] = df[cols[8]].map(gpa_map)
    df["rend_perc"] = df[cols[9]].map(perf_map)
    df["sleep_hours"] = df[cols[18]].map(sleep_hours_map)
    df["sleep_adequacy"] = df[cols[18]].map(sleep_adequacy_map)

    for q in range(10, 26):
        df[f"q{q}"] = pd.to_numeric(df[cols[q]], errors="coerce")
    df["q15_inv"] = 6 - df["q15"]
    df["q20_inv"] = 6 - df["q20"]
    df["q21_inv"] = 6 - df["q21"]
    df["q22_inv"] = 6 - df["q22"]

    df["GTL"] = df[["q10", "q11", "q12", "q13", "q14"]].mean(axis=1)
    df["CalSueno"] = df[["q16", "q17", "sleep_adequacy", "q19", "q21_inv"]].mean(axis=1)
    df["OcioActivo"] = df["q23"]
    df["OcioDigital"] = df["q24"]
    df["Procrastinacion"] = df["q15"]
    df["InfluenciaPercibida"] = df["q25"]
    return df


def add_constant(data: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    return sm.add_constant(data[x_cols], has_constant="add")


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def regression_table(model, data: pd.DataFrame, y_col: str) -> pd.DataFrame:
    y_std = data[y_col].std(ddof=1)
    rows = []
    for name in model.params.index:
        beta = np.nan
        if name != "const":
            beta = model.params[name] * data[name].std(ddof=1) / y_std
        rows.append({
            "Predictor": name,
            "B": model.params[name],
            "SE": model.bse[name],
            "t": model.tvalues[name],
            "p": model.pvalues[name],
            "Beta_estandarizado": beta,
        })
    return pd.DataFrame(rows)


def generate_figures(data: pd.DataFrame, ols_model, exp_model, x_cols: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = add_constant(data, x_cols)
    y = data["gpa"]
    pred_ols = ols_model.predict(X)
    pred_exp = np.exp(exp_model.predict(X))

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    ax.scatter(data["GTL"], y, alpha=0.65)
    z = np.polyfit(data["GTL"], y, 1)
    xx = np.linspace(data["GTL"].min(), data["GTL"].max(), 100)
    ax.plot(xx, np.polyval(z, xx), linewidth=2)
    ax.set_xlabel("Gestión del tiempo libre (1-5)")
    ax.set_ylabel("Índice académico recodificado")
    ax.set_title("Dispersión entre gestión del tiempo libre e índice académico")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_dispersion_gtl_gpa.png")
    plt.close(fig)

    for filename, pred, title, ylabel in [
        ("fig2_reales_estimados_ols.png", pred_ols, "Valores reales vs. estimados: regresión lineal múltiple", "GPA estimado por OLS"),
        ("fig3_reales_estimados_exponencial.png", pred_exp, "Valores reales vs. estimados: regresión exponencial", "GPA estimado por modelo exponencial"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
        ax.scatter(y, pred, alpha=0.7)
        lo = min(y.min(), pred.min()) - 0.1
        hi = max(y.max(), pred.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
        ax.set_xlabel("GPA real")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    sample = data.copy()
    sample["pred_ols"] = pred_ols
    sample["pred_exp"] = pred_exp
    sample = sample.sort_values("gpa").reset_index(drop=True).iloc[:40]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ax.plot(range(len(sample)), sample["gpa"], marker="o", linewidth=1, label="GPA real")
    ax.plot(range(len(sample)), sample["pred_ols"], marker="o", linewidth=1, label="OLS")
    ax.plot(range(len(sample)), sample["pred_exp"], marker="o", linewidth=1, label="Exponencial")
    ax.set_xlabel("Observaciones ordenadas por GPA")
    ax.set_ylabel("Índice académico")
    ax.set_title("Comparación entre valores reales y estimados")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_comparacion_reales_estimados.png")
    plt.close(fig)

    coefs = regression_table(ols_model, data, "gpa")
    coefs = coefs[coefs["Predictor"] != "const"].copy()
    coefs["ci_low"] = coefs["B"] - 1.96 * coefs["SE"]
    coefs["ci_high"] = coefs["B"] + 1.96 * coefs["SE"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    y_pos = np.arange(len(coefs))
    ax.errorbar(coefs["B"], y_pos, xerr=[coefs["B"] - coefs["ci_low"], coefs["ci_high"] - coefs["B"]], fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["GTL", "Calidad sueño", "Horas sueño", "Trabaja", "Sexo masculino", "Edad", "Año académico"])
    ax.set_xlabel("Coeficiente B")
    ax.set_title("Coeficientes del modelo lineal múltiple con IC 95%")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_coeficientes_ols.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta del CSV exportado desde Google Forms")
    parser.add_argument("--out", default="results", help="Directorio de salida")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.csv)
    data_all = recode_dataframe(raw)

    x_cols = ["GTL", "CalSueno", "sleep_hours", "trabaja", "sex_male", "edad_num", "year_num"]
    model_data = data_all[["gpa"] + x_cols].dropna().copy()

    X = add_constant(model_data, x_cols)
    y = model_data["gpa"]
    ols_model = sm.OLS(y, X).fit()
    exp_model = sm.OLS(np.log(y), X).fit()
    pred_ols = ols_model.predict(X)
    pred_exp = np.exp(exp_model.predict(X))

    result = {
        "n_raw": int(len(raw)),
        "n_valid_consent": int(len(data_all)),
        "n_gpa_available": int(data_all["gpa"].notna().sum()),
        "n_model": int(len(model_data)),
        "alpha_gtl": cronbach_alpha(data_all[["q10", "q11", "q12", "q13", "q14"]]),
        "alpha_cal_sueno": cronbach_alpha(data_all[["q16", "q17", "sleep_adequacy", "q19", "q21_inv"]]),
        "ols": {"r2": float(ols_model.rsquared), "r2_adj": float(ols_model.rsquared_adj), "f": float(ols_model.fvalue), "p": float(ols_model.f_pvalue), "metrics": metrics(y, pred_ols)},
        "exponential_loglinear": {"r2_log": float(exp_model.rsquared), "r2_original_scale": metrics(y, pred_exp)["R2"], "r2_adj_log": float(exp_model.rsquared_adj), "f": float(exp_model.fvalue), "p": float(exp_model.f_pvalue), "metrics": metrics(y, pred_exp)},
    }

    (out_dir / "resultados_modelo.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    regression_table(ols_model, model_data, "gpa").to_csv(out_dir / "coeficientes_ols.csv", index=False)

    exp_data = model_data.copy()
    exp_data["log_gpa"] = np.log(exp_data["gpa"])
    regression_table(exp_model, exp_data, "log_gpa").to_csv(out_dir / "coeficientes_exponencial_log.csv", index=False)

    desc_cols = ["gpa", "rend_perc", "GTL", "CalSueno", "free_hours", "sleep_hours", "OcioActivo", "OcioDigital", "Procrastinacion", "InfluenciaPercibida", "year_num", "edad_num"]
    data_all[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].to_csv(out_dir / "descriptivos.csv")

    for name, col in {"sexo": "sexo", "edad": raw.columns[2], "anio": raw.columns[5], "trabajo": raw.columns[6], "gpa_categoria": raw.columns[8]}.items():
        counts = data_all[col].value_counts(dropna=False).rename_axis("categoria").reset_index(name="n")
        counts["porcentaje"] = counts["n"] / len(data_all) * 100
        counts.to_csv(out_dir / f"perfil_{name}.csv", index=False)

    generate_figures(model_data, ols_model, exp_model, x_cols, out_dir / "figures")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
