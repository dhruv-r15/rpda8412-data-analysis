# =============================================================================
# RPDA8411 Research Report - Full analysis + figures
# Predictive Analytics for Proactive TB Resource Allocation, Eastern Cape
# =============================================================================
import os, sys, json, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr

try:
    import shap
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"]); import shap
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier
    HAS_XGB = True

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, LeaveOneOut,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

IN_COLAB = "google.colab" in sys.modules
DATA_DIR = "."
OUT = "report_outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
RED, YEL, GRN, BLUE, GREY = "#C0392B", "#E1B12C", "#27AE60", "#2E5A8C", "#7F8C8D"

def finish(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    if IN_COLAB:
        plt.show()
    plt.close(fig)
    print("saved", name)

# ----------------------------------------------------------------------------- data
df = pd.read_csv(f"{DATA_DIR}/facility_panel_clean.csv").drop_duplicates(["facility", "year"]).reset_index(drop=True)
naat = pd.read_csv(f"{DATA_DIR}/district_naat_targets_2026.csv")
prov = pd.read_csv(f"{DATA_DIR}/province_trend_2021_2025.csv").drop_duplicates("indicator").set_index("indicator")

NUM = ["windows_open_pct", "posters_pct", "separate_room_pct", "waiting_min", "enough_room_pct", "tb_screened_pct"]
LAB = {"windows_open_pct": "Windows open (%)", "posters_pct": "TB IC posters displayed (%)",
       "separate_room_pct": "Separate space for coughing patients (%)", "waiting_min": "Waiting time (minutes)",
       "enough_room_pct": "Enough room in waiting area (%)", "tb_screened_pct": "Patients screened for TB (%)"}
DISTRICTS = sorted(df["district"].unique())
X = df[NUM + ["district"]]
y = df["target_high_risk"].astype(int)
N, N_RED, N_YEL = len(df), int(y.sum()), int((1 - y).sum())
R = {"n": N, "n_red": N_RED, "n_yellow": N_YEL, "n_facilities": int(df.facility.nunique()),
     "n_both_years": int((df.groupby("facility").year.nunique() == 2).sum()),
     "n_2023": int((df.year == 2023).sum()), "n_2024": int((df.year == 2024).sum())}
print(R)

def pipe(clf, nums=NUM, cats=("district",)):
    tf = []
    if nums: tf.append(("num", StandardScaler(), list(nums)))
    if cats: tf.append(("cat", OneHotEncoder(categories=[DISTRICTS], handle_unknown="ignore"), list(cats)))
    return Pipeline([("pre", ColumnTransformer(tf)), ("clf", clf)])

def models():
    m = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
         "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
         "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                                   colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                                   scale_pos_weight=N_YEL / N_RED, verbosity=0)}
    return m

def mets(yt, yp, pr=None):
    d = {"accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
         "precision": precision_score(yt, yp, zero_division=0), "recall": recall_score(yt, yp, zero_division=0),
         "specificity": recall_score(yt, yp, pos_label=0, zero_division=0), "f1": f1_score(yt, yp, zero_division=0)}
    if pr is not None and len(set(yt)) == 2:
        d["auc"] = roc_auc_score(yt, pr)
    return d

# ----------------------------------------------------------------------------- 1. descriptives
desc = {}
red_, yel_ = df[df.score == "RED"], df[df.score == "YELLOW"]
for f in NUM:
    u, p = mannwhitneyu(red_[f], yel_[f], alternative="two-sided")
    desc[f] = {"overall_mean": df[f].mean(), "overall_sd": df[f].std(),
               "red_mean": red_[f].mean(), "red_sd": red_[f].std(), "red_median": red_[f].median(),
               "yellow_mean": yel_[f].mean(), "yellow_sd": yel_[f].std(), "yellow_median": yel_[f].median(),
               "p_mwu": p, "uni_auc": u / (len(red_) * len(yel_)),
               "mean_2023": df[df.year == 2023][f].mean(), "mean_2024": df[df.year == 2024][f].mean()}
R["descriptives"] = desc
ct = pd.crosstab([df.district], [df.year, df.score])
R["district_year_counts"] = {d: {f"{yr}_{s}": int(ct.loc[d].get((yr, s), 0)) for yr in (2023, 2024) for s in ("RED", "YELLOW")} for d in ct.index}
R["year_counts"] = {str(yr): {"RED": int(((df.year == yr) & (df.score == "RED")).sum()), "YELLOW": int(((df.year == yr) & (df.score == "YELLOW")).sum())} for yr in (2023, 2024)}

# ----------------------------------------------------------------------------- 2. repeated stratified CV
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
splits = list(rskf.split(X, y))
def one_fit(clf, tr, te, nums=NUM, cats=("district",)):
    p = pipe(clone(clf), nums, cats); p.fit(X.iloc[tr], y.iloc[tr])
    pr = p.predict_proba(X.iloc[te])[:, 1]; yp = p.predict(X.iloc[te])
    return mets(y.iloc[te].values, yp, pr)
def summarise(rows):
    d = pd.DataFrame(rows); return {k: {"mean": d[k].mean(), "sd": d[k].std()} for k in d.columns}
cv = {}
for name, clf in models().items():
    cv[name] = summarise(Parallel(n_jobs=-1)(delayed(one_fit)(clf, tr, te) for tr, te in splits))
cv["District-only logistic baseline"] = summarise(Parallel(n_jobs=-1)(
    delayed(one_fit)(LogisticRegression(max_iter=1000, class_weight="balanced"), tr, te, [], ("district",)) for tr, te in splits))
maj = y.mean()
cv["Majority-class baseline (always RED)"] = {"accuracy": {"mean": maj, "sd": 0}, "balanced_accuracy": {"mean": 0.5, "sd": 0},
    "precision": {"mean": maj, "sd": 0}, "recall": {"mean": 1.0, "sd": 0}, "specificity": {"mean": 0.0, "sd": 0},
    "f1": {"mean": 2 * maj / (1 + maj), "sd": 0}, "auc": {"mean": 0.5, "sd": 0}}
R["repeated_cv"] = cv
from sklearn.model_selection import StratifiedGroupKFold
gsplits = []
for _r in range(20):
    gsplits += list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_r).split(X, y, df["facility"].values))
gcv = {name: summarise(Parallel(n_jobs=-1)(delayed(one_fit)(clf, tr, te) for tr, te in gsplits)) for name, clf in models().items()}
R["grouped_cv"] = gcv
print("grouped CV done")
print("repeated CV done")

# ----------------------------------------------------------------------------- 3. single hold-out (replicates run_models.py split)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ho = {}
for name, clf in models().items():
    p = pipe(clone(clf)).fit(Xtr, ytr)
    pr = p.predict_proba(Xte)[:, 1]; yp = p.predict(Xte)
    ho[name] = {**mets(yte.values, yp, pr), "confusion": confusion_matrix(yte, yp).tolist()}
R["holdout"] = {"train_n": len(ytr), "test_n": len(yte), "test_red": int(yte.sum()), "test_yellow": int((1 - yte).sum()), "models": ho}

# ----------------------------------------------------------------------------- 4. out-of-fold predictions (ROC / confusion)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof, oof_m, roc = {}, {}, {}
for name, clf in models().items():
    pr = cross_val_predict(pipe(clone(clf)), X, y, cv=skf, method="predict_proba")[:, 1]
    oof[name] = pr; yp = (pr >= 0.5).astype(int)
    oof_m[name] = {**mets(y.values, yp, pr), "confusion": confusion_matrix(y, yp).tolist()}
    fpr, tpr, _ = roc_curve(y, pr); roc[name] = (fpr, tpr)
R["oof"] = oof_m

# ----------------------------------------------------------------------------- 5. temporal validation: train 2023 -> test 2024
tr, te = np.where(df.year == 2023)[0], np.where(df.year == 2024)[0]
R["temporal"] = {"train_n": len(tr), "test_n": len(te), "test_red": int(y.iloc[te].sum()), "test_yellow": int((1 - y.iloc[te]).sum()),
                 "models": {n: {**one_fit(c, tr, te), "baseline_acc": float(y.iloc[te].mean())} for n, c in models().items()}}
# ----------------------------------------------------------------------------- 6. leave-one-district-out (numeric features only)
lodo = {}
for name, clf in models().items():
    probs = np.zeros(N); per = {}
    for d in DISTRICTS:
        te = np.where(df.district == d)[0]; tr = np.where(df.district != d)[0]
        p = pipe(clone(clf), NUM, None).fit(X.iloc[tr], y.iloc[tr])
        pr = p.predict_proba(X.iloc[te])[:, 1]; probs[te] = pr
        yp = (pr >= 0.5).astype(int); yt = y.iloc[te].values
        per[d] = {"n": len(te), "accuracy": accuracy_score(yt, yp), "red_share": float(yt.mean())}
    yp = (probs >= 0.5).astype(int)
    lodo[name] = {**mets(y.values, yp, probs), "per_district": per}
R["lodo"] = lodo

# ----------------------------------------------------------------------------- 7. facility-level longitudinal (2023 indicators -> 2024 status)
w23 = df[df.year == 2023].set_index("facility"); w24 = df[df.year == 2024].set_index("facility")
common = sorted(set(w23.index) & set(w24.index))
a, b = w23.loc[common], w24.loc[common]
trans = pd.crosstab(a.score, b.score)
R["longitudinal"] = {"n_pairs": len(common),
    "transitions": {f"{i}->{j}": int(trans.loc[i, j]) if (i in trans.index and j in trans.columns) else 0 for i in ("RED", "YELLOW") for j in ("RED", "YELLOW")},
    "persistence_accuracy": float((a.score.values == b.score.values).mean()),
    "red_2024_share": float((b.score == "RED").mean())}
Xl, yl = a[NUM], (b.score == "RED").astype(int)
for nm, clf in [("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))]:
    pr = cross_val_predict(pipe(clone(clf), NUM, None), Xl, yl, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    R["longitudinal"][nm] = mets(yl.values, (pr >= 0.5).astype(int), pr)

# ----------------------------------------------------------------------------- 8. logistic regression odds ratios (bootstrap CI)
lr_full = pipe(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X, y)
names = [n.split("__")[1].replace("district_", "District: ") for n in lr_full.named_steps["pre"].get_feature_names_out()]
coef = lr_full.named_steps["clf"].coef_[0]
rng = np.random.default_rng(42); boots = []
for _ in range(1000):
    idx = rng.integers(0, N, N)
    if y.iloc[idx].nunique() < 2: continue
    boots.append(pipe(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X.iloc[idx], y.iloc[idx]).named_steps["clf"].coef_[0])
boots = np.array(boots)
lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
R["lr_or"] = [{"feature": LAB.get(n, n), "coef": c, "or": np.exp(c), "or_lo": np.exp(l), "or_hi": np.exp(h)}
              for n, c, l, h in zip(names, coef, lo, hi)]

# ----------------------------------------------------------------------------- 9. permutation importance (RF, held-out folds)
perm = {f: [] for f in NUM + ["district"]}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
    p = pipe(clone(models()["Random Forest"])).fit(X.iloc[tr], y.iloc[tr])
    pi = permutation_importance(p, X.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=20, random_state=7)
    for f, m in zip(X.columns, pi.importances_mean): perm[f].append(m)
R["perm_importance"] = {f: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for f, v in perm.items()}

# ----------------------------------------------------------------------------- 10. SHAP (RF, full panel)
pre = pipe(None).named_steps["pre"]
Xt = pre.fit_transform(X); Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
fn = [n.split("__")[1].replace("district_", "District = ") for n in pre.get_feature_names_out()]
pretty = [LAB.get(n, n) for n in fn]
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced").fit(Xt, y)
sv = shap.TreeExplainer(rf).shap_values(Xt)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
msh = np.abs(sv).mean(0); order = np.argsort(-msh)
R["shap"] = [{"feature": pretty[i], "mean_abs_shap": float(msh[i]),
              "direction_rho": float(spearmanr(Xt[:, i], sv[:, i])[0]) if i < len(NUM) else None} for i in order]

# ----------------------------------------------------------------------------- 11. district analysis
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)
dist = df.groupby("district").agg(n=("score", "size"), red=("target_high_risk", "sum")).reset_index()
dist["red_rate"] = dist.red / dist.n
dist[["ci_lo", "ci_hi"]] = [wilson(k, n) for k, n in zip(dist.red, dist.n)]
dmerge = naat.merge(dist, on="district", how="left")
mm = lambda s: (s - s.min()) / (s.max() - s.min())
d24 = df[df.year == 2024].groupby("district").agg(n=("score", "size"), red=("target_high_risk", "sum"))
d23 = df[df.year == 2023].groupby("district").agg(n=("score", "size"), red=("target_high_risk", "sum"))
R["district_year_rates"] = {d: {"2023": (None if d not in d23.index else {"n": int(d23.loc[d, "n"]), "red": int(d23.loc[d, "red"]), "rate": float(d23.loc[d, "red"] / d23.loc[d, "n"])}),
                                "2024": (None if d not in d24.index else {"n": int(d24.loc[d, "n"]), "red": int(d24.loc[d, "red"]), "rate": float(d24.loc[d, "red"] / d24.loc[d, "n"])})} for d in DISTRICTS}
def composite(w_naat=0.5, impute=None, naat_only=False, red_series=None):
    red = dmerge["red_rate"].copy() if red_series is None else red_series.copy()
    if impute is not None: red = red.fillna(impute)
    nn, rn = mm(dmerge["naat_required_2026_27"]), mm(red)
    if naat_only: return nn
    return pd.Series(np.where(rn.notna(), w_naat * nn + (1 - w_naat) * rn.fillna(0), nn), index=dmerge.index)
scen = {"Base: 50/50, NAAT-only where no panel": composite(0.5), "NAAT weight 25%": composite(0.25),
        "NAAT weight 75%": composite(0.75), "Pooled RED rate imputed (50/50)": composite(0.5, impute=N_RED / N),
        "NAAT target only": composite(naat_only=True),
        "2024 panel only (50/50)": composite(0.5, red_series=dmerge["district"].map((d24.red / d24.n)))}
sc_out = {}
base_rank = scen["Base: 50/50, NAAT-only where no panel"].rank(ascending=False)
for s, v in scen.items():
    rk = v.rank(ascending=False, method="first")
    sc_out[s] = {"scores": dict(zip(dmerge.district, v.round(3))), "ranks": dict(zip(dmerge.district, rk.astype(int))),
                 "tiers": dict(zip(dmerge.district, ["High" if r <= 3 else "Moderate" if r <= 6 else "Low" for r in rk])),
                 "spearman_vs_base": float(spearmanr(base_rank, v.rank(ascending=False))[0])}
R["district"] = {"table": dmerge.assign(composite=scen["Base: 50/50, NAAT-only where no panel"]).to_dict("records"),
                 "scenarios": sc_out, "pooled_red_rate": N_RED / N}

with open(f"{OUT}/results.json", "w") as f:
    json.dump(R, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else o.tolist() if isinstance(o, np.ndarray) else str(o))
print("results.json written")

# =============================================================================
# FIGURES
# =============================================================================
# Fig: provincial TB IC trend
yrs = ["2021", "2022", "2023", "2024"]
tot = prov.loc["tb_ic_facilities_total_assessed", yrs].astype(float)
g, yl_, r_ = [prov.loc[k, yrs].astype(float) / tot * 100 for k in ("tb_ic_facilities_green", "tb_ic_facilities_yellow", "tb_ic_facilities_red")]
fig, ax = plt.subplots(figsize=(6.5, 3.8)); x = np.arange(4)
ax.bar(x, r_, 0.6, color=RED, label="RED (3+ measures failing)")
ax.bar(x, yl_, 0.6, bottom=r_, color=YEL, label="YELLOW (1-2 failing)")
ax.bar(x, g, 0.6, bottom=r_ + yl_, color=GRN, label="GREEN (all met)")
for i, yr in enumerate(yrs):
    ax.text(i, r_.iloc[i] / 2, f"{r_.iloc[i]:.0f}%", ha="center", color="white", fontweight="bold")
    ax.text(i, 102, f"n={int(tot[yr])}", ha="center", fontsize=8, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(yrs); ax.set_ylim(0, 110); ax.set_ylabel("% of assessed facilities")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8)
finish(fig, "fig_province_tbic_trend.png")

# Fig: provincial service quality context
fig, axs = plt.subplots(1, 3, figsize=(9, 3.0))
xs = [2021, 2022, 2023, 2024]
def row(k): return prov.loc[k, yrs].astype(float).values
axs[0].plot(xs, row("avg_waiting_time_after_opening_hrs"), "o-", color=BLUE); axs[0].set_title("Waiting time after\nfacility opens (hours)", fontsize=9)
axs[1].plot(xs, row("pct_users_feel_safe_waiting"), "o-", color=RED); axs[1].set_title("Users who feel safe\nwhile waiting (%)", fontsize=9)
axs[2].plot(xs, row("pct_facility_managers_report_enough_staff"), "o-", color=BLUE, label="Managers: enough staff")
axs[2].plot(xs, row("pct_users_report_always_enough_staff"), "s--", color=YEL, label="Users: always enough staff")
axs[2].set_title("Staffing perceptions (%)", fontsize=9); axs[2].legend(fontsize=7, frameon=False)
for a_ in axs: a_.set_xticks(xs); a_.tick_params(labelsize=8)
fig.tight_layout(); finish(fig, "fig_province_service_quality.png")

# Fig: panel composition by district and year
fig, ax = plt.subplots(figsize=(7, 3.8)); w = 0.38
for j, yr in enumerate((2023, 2024)):
    xx = np.arange(len(DISTRICTS)) + (j - 0.5) * w
    rr = [R["district_year_counts"][d][f"{yr}_RED"] for d in DISTRICTS]; yy = [R["district_year_counts"][d][f"{yr}_YELLOW"] for d in DISTRICTS]
    ax.bar(xx, rr, w, color=RED, alpha=1 if yr == 2024 else 0.6, label=f"RED {yr}")
    ax.bar(xx, yy, w, bottom=rr, color=YEL, alpha=1 if yr == 2024 else 0.6, label=f"YELLOW {yr}")
ax.set_xticks(np.arange(len(DISTRICTS))); ax.set_xticklabels(DISTRICTS, rotation=15); ax.set_ylabel("Facility-years")
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.text(0.01, 0.95, "Left bar = 2023 (lighter), right bar = 2024", transform=ax.transAxes, fontsize=7, color=GREY)
finish(fig, "fig_panel_by_district_year.png")

# Fig: predictor distributions by class
fig, axs = plt.subplots(2, 3, figsize=(9, 5.2))
for a_, f in zip(axs.ravel(), NUM):
    data = [yel_[f].values, red_[f].values]
    bp = a_.boxplot(data, labels=["YELLOW", "RED"], patch_artist=True, widths=0.5, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], (YEL, RED)): patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, d in enumerate(data):
        a_.scatter(np.random.default_rng(1).normal(i + 1, 0.05, len(d)), d, s=6, color="black", alpha=0.35, zorder=3)
    a_.set_title(LAB[f], fontsize=8)
fig.tight_layout(); finish(fig, "fig_predictor_distributions.png")

# Fig: ROC
fig, ax = plt.subplots(figsize=(5, 4.4))
for (nm, (fpr, tpr)), c in zip(roc.items(), (BLUE, RED, GRN)):
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{nm} (AUC = {oof_m[nm]['auc']:.2f})")
ax.plot([0, 1], [0, 1], "--", color=GREY); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", fontsize=8, frameon=False); finish(fig, "fig_roc_curves.png")

# Fig: confusion matrices (out-of-fold)
fig, axs = plt.subplots(1, 3, figsize=(9, 3.1))
for a_, nm in zip(axs, oof_m):
    cm = np.array(oof_m[nm]["confusion"]); a_.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a_.set_xticks([0, 1]); a_.set_xticklabels(["YELLOW", "RED"]); a_.set_yticks([0, 1]); a_.set_yticklabels(["YELLOW", "RED"])
    a_.set_xlabel("Predicted"); a_.set_ylabel("Actual"); a_.set_title(nm, fontsize=9)
fig.tight_layout(); finish(fig, "fig_confusion_matrices.png")

# Fig: validation scheme comparison (AUC)
schemes = {"Repeated 5-fold CV\n(random split)": {n: cv[n]["auc"]["mean"] for n in models()},
           "Grouped CV\n(facility held out)": {n: gcv[n]["auc"]["mean"] for n in models()},
           "Out-of-time\n(train 2023, test 2024)": {n: R["temporal"]["models"][n]["auc"] for n in models()},
           "Leave-one-district-out\n(no district feature)": {n: lodo[n]["auc"] for n in models()}}
fig, ax = plt.subplots(figsize=(8.6, 3.9)); w = 0.26
for j, (nm, c) in enumerate(zip(models(), (BLUE, RED, GRN))):
    vals = [schemes[s][nm] for s in schemes]
    bars = ax.bar(np.arange(len(schemes)) + (j - 1) * w, vals, w, color=c, label=nm)
    for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax.axhline(0.5, color=GREY, ls="--", lw=1); ax.text(len(schemes) - 0.55, 0.505, "chance", fontsize=7, color=GREY, ha="right")
ax.set_xticks(range(len(schemes))); ax.set_xticklabels(list(schemes), fontsize=8); ax.set_ylim(0, 1.12); ax.set_ylabel("AUC-ROC")
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2)); finish(fig, "fig_validation_comparison.png")

# Fig: SHAP bar + beeswarm
top = R["shap"][:10][::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top], color=[RED if t["mean_abs_shap"] > 0.1 else BLUE for t in top])
for i, t in enumerate(top): ax.text(t["mean_abs_shap"] + 0.003, i, f"{t['mean_abs_shap']:.3f}", va="center", fontsize=8)
ax.set_xlabel("Mean |SHAP value| (impact on predicted probability of RED)"); finish(fig, "fig_shap_bar.png")
plt.figure(figsize=(7.2, 4.4))
shap.summary_plot(sv, Xt, feature_names=pretty, show=False, max_display=10, plot_size=None)
finish(plt.gcf(), "fig_shap_beeswarm.png")

# Fig: LR odds ratios
orr = [o for o in R["lr_or"] if not o["feature"].startswith("District")]
fig, ax = plt.subplots(figsize=(7, 3.6)); yy = np.arange(len(orr))[::-1]
for yv, o in zip(yy, orr):
    ax.plot([o["or_lo"], o["or_hi"]], [yv, yv], color=BLUE, lw=2); ax.plot(o["or"], yv, "o", color=RED)
ax.axvline(1, color=GREY, ls="--"); ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels([o["feature"] for o in orr], fontsize=8)
ax.set_xlabel("Odds ratio for RED per +1 SD (log scale, 95% bootstrap CI)"); finish(fig, "fig_lr_odds_ratios.png")

# Fig: permutation importance
pi = sorted(R["perm_importance"].items(), key=lambda kv: kv[1]["mean"])
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.barh([LAB.get(k, "District") for k, _ in pi], [v["mean"] for _, v in pi], xerr=[v["sd"] for _, v in pi], color=BLUE, ecolor=GREY)
ax.set_xlabel("Mean decrease in held-out AUC when feature is permuted"); finish(fig, "fig_permutation_importance.png")

# Fig: district RED rates with Wilson CI
dd = dist.sort_values("red_rate")
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.barh(dd.district, dd.red_rate * 100, color=RED, alpha=0.8)
ax.errorbar(dd.red_rate * 100, dd.district, xerr=[(dd.red_rate - dd.ci_lo) * 100, (dd.ci_hi - dd.red_rate) * 100], fmt="none", ecolor="black", capsize=3)
for i, (r_, n_) in enumerate(zip(dd.red_rate, dd.n)): ax.text(2, i, f"n={n_}", va="center", color="white", fontsize=8)
ax.set_xlabel("% of facility-years scoring RED (95% Wilson CI)"); ax.set_xlim(0, 105); finish(fig, "fig_district_red_rates.png")

# Fig: district composite ranking
base = R["district"]["table"]; bd = pd.DataFrame(base).sort_values("composite")
tier_of = R["district"]["scenarios"]["Base: 50/50, NAAT-only where no panel"]["tiers"]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(bd.district, bd.composite, color=[{"High": RED, "Moderate": YEL, "Low": GRN}[tier_of[d]] for d in bd.district])
for i, (v, rr) in enumerate(zip(bd.composite, bd.red_rate)): ax.text(v + 0.01, i, f"{v:.3f}" + ("" if pd.notna(rr) else "*"), va="center", fontsize=8)
ax.set_xlabel("Composite resource-demand score (* = testing-burden component only)"); ax.set_xlim(0, 0.8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="High"), Patch(color=YEL, label="Moderate"), Patch(color=GRN, label="Low")], loc="lower right", frameon=False, title="Tier")
finish(fig, "fig_district_ranking.png")

# Fig: ranking sensitivity (rank per scenario)
fig, ax = plt.subplots(figsize=(7.4, 3.9))
sn = list(scen); cols = plt.cm.tab10(np.linspace(0, 1, 8))
for c, d in zip(cols, DISTRICTS + [x for x in naat.district if x not in DISTRICTS]):
    ax.plot(range(len(sn)), [sc_out[s]["ranks"][d] for s in sn], "o-", color=c, label=d)
ax.set_xticks(range(len(sn))); ax.set_xticklabels(["Base", "NAAT 25%", "NAAT 75%", "Imputed\nRED rate", "NAAT\nonly", "2024\nonly"], fontsize=8)
ax.invert_yaxis(); ax.set_yticks(range(1, 9)); ax.set_ylabel("Rank (1 = highest demand)")
ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left"); finish(fig, "fig_ranking_sensitivity.png")

# Fig: pipeline diagram
fig, ax = plt.subplots(figsize=(9, 5.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def box(x, y_, w_, h_, t, fc="#EAF0F8", ec=BLUE):
    ax.add_patch(plt.Rectangle((x, y_), w_, h_, fc=fc, ec=ec, lw=1.4)); ax.text(x + w_ / 2, y_ + h_ / 2, t, ha="center", va="center", fontsize=8)
def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
box(3.0, 8.7, 4.0, 1.0, "Ritshidze EC State of Health reports\n2023 + 2024: facility TB IC scorecards\n(n = 93 facility-years, 5 districts)", "#F5F5F5", GREY)
ax.text(0.3, 9.25, "A: FACILITY CLASSIFIER", fontsize=9, fontweight="bold", color=BLUE)
ax.text(7.3, 9.25, "B: DISTRICT RANKING", fontsize=9, fontweight="bold", color=BLUE)
A = ["Clean, harmonise, complete-case\n(2024 windows category -> %)", "LR / RF / XGBoost\n(class-weighted)", "Validation: repeated CV,\nout-of-time, leave-district-out", "SHAP, permutation importance,\nodds ratios", "Facility TB IC risk flags"]
B = ["NDoH End TB Plan 1.0, Annexure C\nNAAT targets, all 8 districts", "% facilities RED per district\n(Wilson 95% CI)", "Min-max normalise,\ncombine (50/50)", "Sensitivity: weights, imputation,\nNAAT-only", "District resource-demand tier"]
for i, (ta, tb) in enumerate(zip(A, B)):
    yb = 6.7 - i * 1.5
    last = i == len(A) - 1
    box(0.3, yb, 4.2, 1.0, ta, "#FDEDEC" if last else "#EAF0F8", RED if last else BLUE)
    box(5.5, yb, 4.2, 1.0, tb, "#FDEDEC" if last else "#EAF0F8", RED if last else BLUE)
    if i < len(A) - 1: arr(2.4, yb, 2.4, yb - 0.5); arr(7.6, yb, 7.6, yb - 0.5)
arr(4.2, 8.7, 2.4, 7.7); arr(5.8, 8.7, 7.6, 7.7)
finish(fig, "fig_pipeline.png")
print("ALL DONE")
