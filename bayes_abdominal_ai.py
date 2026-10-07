#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayes_abdominal_ai.py
=====================================================================
Hierarchical Bayesian parameter estimation and population calibration of an
invariant-based ANGULAR-INTEGRATION (AI) hyperelastic model for abdominal soft
tissues (abdominal aorta / AAA wall, linea alba, rectus sheath, ...), with
patient covariates (age, sex, smoking history).

Goal: replace patient-specific fitting by a covariate-informed population
model.  For a NEW patient you only need (tissue, age, sex, smoking) and the
script returns a full posterior-predictive parameter set + stress bands.

Constitutive model (incompressible, in-plane dispersed collagen):
    W = mu/2 (I1 - 3) + int_0^pi rho(theta; phi, b) psi_f(I4(theta)) dtheta
    I4(theta) = lam1^2 cos^2 theta + lam2^2 sin^2 theta
    psi_f     = k1/(2 k2) [exp(k2 (I4-1)^2) - 1]      (fibres in tension only)
    rho       = bimodal pi-periodic von Mises with half-angle phi, concentration b
Cauchy stresses (plane stress, sigma33 = 0 fixes the pressure):
    s11 = mu(lam1^2 - lam3^2) + 2 int rho psi_f' lam1^2 cos^2 theta dtheta
    s22 = mu(lam2^2 - lam3^2) + 2 int rho psi_f' lam2^2 sin^2 theta dtheta
Uniaxial tests: lam2 is solved inside the graph from s22 = 0 (unrolled Newton),
so the whole likelihood is differentiable and NUTS can be used.

Hierarchical regression (per parameter p, sample j, subject s(j), tissue t(j)):
    log theta_{j,p} = alpha_{t,p} + beta_{age,p} age_z + beta_{smk,p} smoker
                      + beta_{male,p} male + tau_p z_{s,p},   z ~ N(0,1)
    sigma_obs ~ StudentT(nu=4, model, s_abs + s_rel |model|)

DATA SOURCES (see --source):
  synthetic : virtual cohort with KNOWN ground truth (pipeline verification)
  mendeley  : open Mendeley Data uniaxial human abdominal aorta datasets
              (normal-diameter: yfj4wfszbw, AAA: x64srrc39p). Downloaded via
              the public API when reachable, otherwise put the "Download All"
              zip contents into data/raw/<dataset_id>/ and re-run.
              Covariates (age/sex/smoking) must be supplied in a metadata CSV
              (--meta) - they are NOT inside the raw curve files.
  csv       : your own / digitised literature curves in the long format
              described in load_long_csv().

Usage examples
  python bayes_abdominal_ai.py --source synthetic --draws 1000 --tune 1000
  python bayes_abdominal_ai.py --source mendeley --meta data/meta.csv
  python bayes_abdominal_ai.py --source csv --csv mydata.csv
Requirements: numpy scipy pandas matplotlib pymc>=5 arviz requests openpyxl xlrd
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import i0

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*object mode.*")

PARAMS = ["mu", "k1", "k2", "b"]            # log-modelled parameters
PARAM_TEX = {"mu": r"$\mu$ [kPa]", "k1": r"$k_1$ [kPa]",
             "k2": r"$k_2$ [-]", "b": r"$b$ (dispersion conc.)"}
PRIOR_LOGMEAN = {"mu": np.log(20.0), "k1": np.log(100.0),
                 "k2": np.log(8.0), "b": np.log(2.0)}
COVARIATES_ALL = ["age_z", "smoker", "male"]
NQ = 32                                       # angular quadrature points
RNG = np.random.default_rng(20261007)


# =============================================================================
# 1. Constitutive model (NumPy reference implementation)
# =============================================================================
def gauss_theta(n=NQ):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w      # nodes on [0, pi]


THETA_Q, W_Q = gauss_theta()


def fibre_density(theta, phi, b):
    """Bimodal, pi-periodic von Mises density normalised on [0, pi)."""
    return 0.5 * (np.exp(b * np.cos(2 * (theta - phi))) +
                  np.exp(b * np.cos(2 * (theta + phi)))) / (np.pi * i0(b))


def ai_stress_np(lam1, lam2, mu, k1, k2, b, phi, load_angle=0.0):
    """Cauchy stresses (kPa) of the AI model; arrays broadcast over points."""
    lam1 = np.atleast_1d(lam1)[:, None]
    lam2 = np.atleast_1d(lam2)[:, None]
    th = THETA_Q[None, :]
    rho = fibre_density(th + load_angle, phi, b)
    rho = rho / np.sum(rho * W_Q, axis=1, keepdims=True)
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    e = lam1 ** 2 * c2 + lam2 ** 2 * s2 - 1.0
    dpsi = np.where(e > 0, k1 * e * np.exp(np.minimum(k2 * e ** 2, 50.0)), 0.0)
    lam3sq = 1.0 / (lam1 * lam2) ** 2
    s11 = mu * (lam1 ** 2 - lam3sq) + 2 * np.sum(W_Q * rho * dpsi * lam1 ** 2 * c2, axis=1, keepdims=True)
    s22 = mu * (lam2 ** 2 - lam3sq) + 2 * np.sum(W_Q * rho * dpsi * lam2 ** 2 * s2, axis=1, keepdims=True)
    return s11[:, 0], s22[:, 0]


def uniaxial_np(lam1, mu, k1, k2, b, phi, load_angle=0.0):
    """Uniaxial s11 with lateral stretch solved from s22 = 0."""
    out, lam2s = [], []
    for l1 in np.atleast_1d(lam1):
        f = lambda l2: ai_stress_np(l1, l2, mu, k1, k2, b, phi, load_angle)[1][0]
        try:
            l2 = brentq(f, 0.3, 1.0 + 1e-9)
        except ValueError:
            l2 = 1.0 / np.sqrt(l1)
        lam2s.append(l2)
        out.append(ai_stress_np(l1, l2, mu, k1, k2, b, phi, load_angle)[0][0])
    return np.array(out), np.array(lam2s)


# =============================================================================
# 2. Data acquisition
# =============================================================================
MENDELEY_DATASETS = {
    # id: (version, tissue label)
    "yfj4wfszbw": (4, "abdominal_aorta_normal"),
    "x64srrc39p": (4, "abdominal_aorta_AAA"),
}
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={v}"


def _download_mendeley(ds_id, version, dest):
    """Best-effort download through the Mendeley Data public API (recursive)."""
    try:
        import requests
    except ImportError:
        print("  requests not installed - skipping download");  return False
    os.makedirs(dest, exist_ok=True)
    got = 0

    def walk(folder_id):
        nonlocal got
        url = (f"https://data.mendeley.com/public-api/datasets/{ds_id}/files"
               f"?folder_id={folder_id}&version={version}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        for item in r.json():
            name = item.get("filename", "")
            dl = (item.get("content_details") or {}).get("download_url")
            if dl and name.lower().endswith((".xls", ".xlsx", ".csv", ".txt", ".zip")):
                path = os.path.join(dest, f"{folder_id}_{name}".replace("/", "_"))
                if not os.path.exists(path):
                    with open(path, "wb") as fh:
                        fh.write(requests.get(dl, timeout=120).content)
                got += 1
        # sub-folders
        fr = requests.get(f"https://data.mendeley.com/public-api/datasets/{ds_id}/folders/{version}",
                          timeout=60)
        return fr.json() if fr.ok else []

    try:
        folders = walk("root")
        for fo in folders:
            if fo.get("id"):
                try:
                    walk(fo["id"])
                except Exception:
                    pass
    except Exception as exc:
        print(f"  Mendeley API download failed for {ds_id}: {exc}")
    print(f"  {ds_id}: {got} files available in {dest}")
    return got > 0


_STRAIN_KEYS = ("strain", "deform", "extens", "alonga")
_STRESS_KEYS = ("stress", "tens", "tensão", "tensao")


def _parse_curve_table(df):
    """Heuristically extract (engineering strain, engineering stress) columns."""
    cols = [str(c).lower() for c in df.columns]
    def pick(keys, exclude=None):
        for i, c in enumerate(cols):
            if any(k in c for k in keys) and (exclude is None or i != exclude):
                return i
        return None
    i_e = pick(_STRAIN_KEYS)
    i_s = pick(_STRESS_KEYS, exclude=i_e)
    num = df.apply(pd.to_numeric, errors="coerce")
    if i_e is None or i_s is None:
        good = [c for c in num.columns if num[c].notna().mean() > 0.8]
        if len(good) < 2:
            return None
        i_e, i_s = list(num.columns).index(good[-2]), list(num.columns).index(good[-1])
    eps = num.iloc[:, i_e].to_numpy(float)
    sig = num.iloc[:, i_s].to_numpy(float)
    ok = np.isfinite(eps) & np.isfinite(sig)
    eps, sig = eps[ok], sig[ok]
    if len(eps) < 10:
        return None
    if np.nanmax(eps) > 3.0:                  # given in percent
        eps = eps / 100.0
    unit = "MPa" if "mpa" in cols[i_s] or np.nanmax(sig) < 20 else "kPa"
    sig_kpa = sig * (1000.0 if unit == "MPa" else 1.0)
    return eps, sig_kpa


def _read_any_table(path):
    if path.lower().endswith(".csv") or path.lower().endswith(".txt"):
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep, decimal="," if sep == ";" else ".")
                if df.shape[1] >= 2:
                    return [df]
            except Exception:
                pass
        return []
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return []
    out = []
    for df in sheets.values():
        # header may not be on the first row: find first row with >=2 strings
        for h in range(min(15, len(df))):
            row = df.iloc[h].astype(str).str.lower()
            if row.str.contains("|".join(_STRAIN_KEYS + _STRESS_KEYS)).sum() >= 2:
                df = df.iloc[h + 1:].set_axis(df.iloc[h].astype(str), axis=1)
                break
        out.append(df)
    return out


def preprocess_uniaxial(eps, sig_eng_kpa, frac_of_peak=0.6, n_keep=25):
    """Engineering -> stretch & Cauchy stress, keep pre-damage range only."""
    ip = int(np.argmax(sig_eng_kpa))
    eps, sig = eps[:ip + 1], sig_eng_kpa[:ip + 1]
    mask = (sig <= frac_of_peak * sig_eng_kpa[ip]) & (eps >= 0)
    eps, sig = eps[mask], sig[mask]
    if len(eps) < 6:
        return None
    order = np.argsort(eps)
    eps, sig = eps[order], sig[order]
    lam = 1.0 + eps
    cauchy = sig * lam                          # incompressible uniaxial
    idx = np.unique(np.linspace(0, len(lam) - 1, n_keep).astype(int))
    return lam[idx], cauchy[idx]


def load_mendeley(raw_dir="data/raw", meta_csv=None, try_download=True):
    rows = []
    for ds_id, (ver, tissue) in MENDELEY_DATASETS.items():
        dest = os.path.join(raw_dir, ds_id)
        if try_download and not (os.path.isdir(dest) and os.listdir(dest)):
            print(f"Downloading Mendeley dataset {ds_id} ...")
            _download_mendeley(ds_id, ver, dest)
        if not os.path.isdir(dest):
            continue
        # unpack zips
        for root, _, files in os.walk(dest):
            for fn in files:
                if fn.lower().endswith(".zip"):
                    try:
                        zipfile.ZipFile(os.path.join(root, fn)).extractall(os.path.join(root, fn[:-4]))
                    except Exception:
                        pass
        for root, _, files in os.walk(dest):
            for fn in files:
                if not fn.lower().endswith((".xls", ".xlsx", ".csv")) or "histolog" in fn.lower():
                    continue
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, dest)
                case = re.search(r"(?:case|caso)\s*[_\- ]?([A-Za-z]?\d+|[A-Z])\b", rel, re.I)
                case_id = f"{ds_id}_{case.group(1).upper() if case else os.path.dirname(rel) or fn}"
                for k, df in enumerate(_read_any_table(path)):
                    parsed = _parse_curve_table(df)
                    if parsed is None:
                        continue
                    pp = preprocess_uniaxial(*parsed)
                    if pp is None:
                        continue
                    lam, cs = pp
                    sid = f"{case_id}_{os.path.splitext(fn)[0]}_{k}"
                    for l, s in zip(lam, cs):
                        rows.append(dict(sample_id=sid, subject_id=case_id, tissue=tissue,
                                         protocol="uniaxial", load_angle_deg=0.0,
                                         lam1=l, lam2=np.nan, sig11_kPa=s, sig22_kPa=np.nan))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if meta_csv and os.path.exists(meta_csv):
        meta = pd.read_csv(meta_csv)          # columns: subject_id, age, sex, smoker
        df = df.merge(meta, on="subject_id", how="left")
    else:
        tpl = os.path.join(raw_dir, "meta_template.csv")
        pd.DataFrame(dict(subject_id=sorted(df.subject_id.unique()), age="", sex="", smoker="")
                     ).to_csv(tpl, index=False)
        print(f"  No covariate file given. Fill in {tpl} and pass it with --meta")
    print(f"Mendeley: {df.sample_id.nunique()} curves from {df.subject_id.nunique()} subjects")
    return df


def load_long_csv(path):
    """Long format, one row per data point:
    sample_id, subject_id, tissue, protocol(uniaxial|biaxial), load_angle_deg,
    lam1, lam2 (NaN for uniaxial), sig11_kPa, sig22_kPa (NaN for uniaxial),
    age [yr], sex (M/F), smoker (0/1)"""
    df = pd.read_csv(path)
    need = {"sample_id", "subject_id", "tissue", "protocol", "lam1", "sig11_kPa"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    for c, d in [("load_angle_deg", 0.0), ("lam2", np.nan), ("sig22_kPa", np.nan)]:
        if c not in df:
            df[c] = d
    return df


# ---- synthetic virtual cohort ------------------------------------------------
TRUE = {
    # tissue: (log-mean mu,k1,k2,b), phi_deg, protocol, angles
    "abdominal_aorta": (dict(mu=np.log(25), k1=np.log(60), k2=np.log(8), b=np.log(2.5)), 40.0,
                        "biaxial", [0.0], 1.18),
    "linea_alba":      (dict(mu=np.log(12), k1=np.log(300), k2=np.log(4), b=np.log(3.0)), 80.0,
                        "uniaxial", [0.0, 90.0], 1.12),
    "rectus_sheath":   (dict(mu=np.log(10), k1=np.log(150), k2=np.log(6), b=np.log(1.5)), 60.0,
                        "uniaxial", [0.0, 90.0], 1.15),
}
TRUE_BETA = {   # covariate effects on log-parameters (ground truth for synthetic test)
    "age_z":  dict(mu=0.10, k1=0.25, k2=0.20, b=0.00),
    "smoker": dict(mu=0.00, k1=0.20, k2=0.15, b=-0.15),
    "male":   dict(mu=0.05, k1=0.10, k2=0.00, b=0.00),
}
TRUE_TAU = dict(mu=0.15, k1=0.20, k2=0.15, b=0.15)


def make_synthetic(n_subjects=24, noise_rel=0.05, noise_abs=1.0):
    rows, truth = [], []
    tissues = list(TRUE)
    for s in range(n_subjects):
        age = RNG.uniform(25, 85)
        sex = RNG.choice(["M", "F"])
        smk = int(RNG.random() < 0.4)
        x = dict(age_z=(age - 55) / 15, smoker=smk, male=int(sex == "M"))
        z = {p: RNG.normal() for p in PARAMS}
        for t in RNG.choice(tissues, size=2, replace=False):
            logm, phi, prot, angles, lmax = TRUE[t]
            th = {p: np.exp(logm[p] + sum(TRUE_BETA[c][p] * x[c] for c in COVARIATES_ALL)
                            + TRUE_TAU[p] * z[p]) for p in PARAMS}
            truth.append(dict(subject_id=f"S{s:02d}", tissue=t, **th))
            for ang in angles:
                sid = f"S{s:02d}_{t}_{int(ang)}"
                if prot == "biaxial":
                    for ratio in [(1.0, 1.0), (1.0, 0.75), (0.75, 1.0)]:
                        g = np.linspace(0, 1, 12)[1:]
                        l1 = 1 + (lmax - 1) * ratio[0] * g
                        l2 = 1 + (lmax - 1) * ratio[1] * g
                        s11, s22 = ai_stress_np(l1, l2, th["mu"], th["k1"], th["k2"], th["b"],
                                                np.deg2rad(phi), np.deg2rad(ang))
                        n1 = RNG.standard_t(4, len(l1)) * (noise_abs + noise_rel * np.abs(s11))
                        n2 = RNG.standard_t(4, len(l1)) * (noise_abs + noise_rel * np.abs(s22))
                        for a in range(len(l1)):
                            rows.append(dict(sample_id=f"{sid}_r{ratio[0]}_{ratio[1]}", subject_id=f"S{s:02d}",
                                             tissue=t, protocol="biaxial", load_angle_deg=ang,
                                             lam1=l1[a], lam2=l2[a], sig11_kPa=s11[a] + n1[a],
                                             sig22_kPa=s22[a] + n2[a], age=age, sex=sex, smoker=smk))
                else:
                    l1 = np.linspace(1.0, lmax, 14)[1:]
                    s11, _ = uniaxial_np(l1, th["mu"], th["k1"], th["k2"], th["b"],
                                         np.deg2rad(phi), np.deg2rad(ang))
                    n1 = RNG.standard_t(4, len(l1)) * (noise_abs + noise_rel * np.abs(s11))
                    for a in range(len(l1)):
                        rows.append(dict(sample_id=sid, subject_id=f"S{s:02d}", tissue=t,
                                         protocol="uniaxial", load_angle_deg=ang, lam1=l1[a],
                                         lam2=np.nan, sig11_kPa=s11[a] + n1[a], sig22_kPa=np.nan,
                                         age=age, sex=sex, smoker=smk))
    return pd.DataFrame(rows), pd.DataFrame(truth)


# =============================================================================
# 3. Data container
# =============================================================================
@dataclass
class Prepared:
    df: pd.DataFrame
    samples: pd.DataFrame
    tissues: list
    subjects: list
    covariates: list
    X: np.ndarray                      # (n_samples, n_cov)
    t_idx: np.ndarray                  # tissue index per sample
    s_idx: np.ndarray                  # subject index per sample
    phi_init: dict = field(default_factory=dict)


def prepare(df):
    df = df.copy()
    df["sex"] = df.get("sex", pd.Series(np.nan, index=df.index))
    covs = []
    if "age" in df and df["age"].notna().any():
        df["age_z"] = (df["age"].fillna(df["age"].mean()) - 55.0) / 15.0;  covs.append("age_z")
    if "smoker" in df and df["smoker"].notna().any():
        df["smoker"] = df["smoker"].fillna(0).astype(float);  covs.append("smoker")
    if df["sex"].notna().any():
        df["male"] = (df["sex"].astype(str).str.upper().str[0] == "M").astype(float);  covs.append("male")
    samples = df.groupby("sample_id").first().reset_index()
    tissues = sorted(samples.tissue.unique())
    subjects = sorted(samples.subject_id.unique())
    X = samples[covs].to_numpy(float) if covs else np.zeros((len(samples), 0))
    t_idx = samples.tissue.map({t: i for i, t in enumerate(tissues)}).to_numpy()
    s_idx = samples.subject_id.map({s: i for i, s in enumerate(subjects)}).to_numpy()
    df["j"] = df.sample_id.map({s: i for i, s in enumerate(samples.sample_id)})
    print(f"Prepared: {len(df)} points | {len(samples)} curves | {len(subjects)} subjects | "
          f"tissues={tissues} | covariates={covs}")
    return Prepared(df, samples, tissues, subjects, covs, X, t_idx, s_idx)


# =============================================================================
# 3b. Forward-only lateral-stretch solver (uniaxial tests)
# =============================================================================
def newton_lam2_np(l1, mu, k1, k2, wrho, n_iter=30, tol=1e-12):
    """Solve s22(lam1, lam2) = 0 for lam2 (uniaxial, plane stress). Column inputs,
    wrho = quadrature weights x normalised fibre density, shape (n, NQ)."""
    th = THETA_Q[None, :]
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    l2 = 1.0 / np.sqrt(l1)
    for _ in range(n_iter):
        e = l1 ** 2 * c2 + l2 ** 2 * s2 - 1.0
        ep = np.maximum(e, 0.0)
        ex = np.exp(np.minimum(k2 * ep ** 2, 50.0))
        dp = k1 * ep * ex
        l3 = 1.0 / (l1 * l2) ** 2
        A2 = np.sum(wrho * dp * s2, axis=1, keepdims=True)
        f = mu * (l2 ** 2 - l3) + 2 * l2 ** 2 * A2
        d2 = np.where(e > 0, k1 * ex * (1 + 2 * k2 * ep ** 2), 0.0)
        d = (mu * (2 * l2 + 2 * l3 / l2) + 4 * l2 * A2
             + 4 * l2 ** 3 * np.sum(wrho * d2 * s2 ** 2, axis=1, keepdims=True))
        step = np.clip(f / (d + 1e-12), -0.1, 0.1)
        l2 = np.clip(l2 - step, 0.3, 1.0)
        if np.max(np.abs(step)) < tol:
            break
    return l2


_LSO_CLASS = None


def LateralStretchOp(n_iter=30):
    """PyTensor Op wrapping newton_lam2_np. Its gradient is never used (output is
    wrapped in disconnected_grad); exact sensitivities come from one connected
    Newton step in the graph (implicit-function theorem)."""
    global _LSO_CLASS
    if _LSO_CLASS is None:
        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
        import pytensor.tensor as pt

        class _LSO(Op):
            __props__ = ("n_iter",)

            def __init__(self, n_iter):
                self.n_iter = int(n_iter)

            def make_node(self, *inputs):
                inputs = [pt.as_tensor_variable(x).astype("float64") for x in inputs]
                return Apply(self, inputs, [pt.tensor(dtype="float64", shape=(None, 1))])

            def perform(self, node, inputs, outputs):
                l1, mu, k1, k2, wrho = [np.asarray(x, dtype=float) for x in inputs]
                outputs[0][0] = newton_lam2_np(l1, mu, k1, k2, wrho, self.n_iter)

            def grad(self, inputs, g):
                return [pt.zeros_like(x) for x in inputs]

        _LSO_CLASS = _LSO
    return _LSO_CLASS(n_iter)


# =============================================================================
# 4. Hierarchical PyMC model (differentiable AI model in PyTensor)
# =============================================================================
def build_model(P: Prepared, newton_iter=30):
    import pymc as pm
    import pytensor.tensor as pt

    th = pt.as_tensor_variable(THETA_Q[None, :])
    wq = pt.as_tensor_variable(W_Q[None, :])
    c2, s2 = pt.cos(th) ** 2, pt.sin(th) ** 2

    def density(b, phi, ang):
        # numerically-normalised bimodal von Mises density on the quadrature grid
        rho = 0.5 * (pt.exp(b * (pt.cos(2 * (th + ang - phi)) - 1)) +
                     pt.exp(b * (pt.cos(2 * (th + ang + phi)) - 1)))
        return wq * rho / pt.sum(rho * wq, axis=1, keepdims=True)      # weights folded in

    def stress(l1, l2, mu, k1, k2, wrho, deriv=False):
        """Cauchy stresses; column-vector inputs (n,1). If deriv, also d s22/d lam2."""
        e = l1 ** 2 * c2 + l2 ** 2 * s2 - 1.0
        ep = pt.maximum(e, 0.0)
        ex = pt.exp(pt.minimum(k2 * ep ** 2, 50.0))
        dpsi = k1 * ep * ex
        l3 = 1.0 / (l1 * l2) ** 2
        A2 = pt.sum(wrho * dpsi * s2, axis=1, keepdims=True)
        s11 = mu * (l1 ** 2 - l3) + 2 * l1 ** 2 * pt.sum(wrho * dpsi * c2, axis=1, keepdims=True)
        s22 = mu * (l2 ** 2 - l3) + 2 * l2 ** 2 * A2
        if not deriv:
            return s11, s22
        d2psi = pt.switch(e > 0, k1 * ex * (1 + 2 * k2 * ep ** 2), 0.0)
        ds22 = (mu * (2 * l2 + 2 * l3 / l2) + 4 * l2 * A2 +
                4 * l2 ** 3 * pt.sum(wrho * d2psi * s2 ** 2, axis=1, keepdims=True))
        return s11, s22, ds22

    df = P.df
    nT, nS, nC = len(P.tissues), len(P.subjects), len(P.covariates)
    coords = {"tissue": P.tissues, "subject": P.subjects, "param": PARAMS,
              "cov": P.covariates, "specimen": P.samples.sample_id.tolist()}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=np.array([PRIOR_LOGMEAN[p] for p in PARAMS])[None, :],
                          sigma=1.0, dims=("tissue", "param"))
        if nC:
            beta = pm.Normal("beta", 0.0, 0.5, dims=("cov", "param"))
        tau = pm.HalfNormal("tau", 0.3, dims="param")
        z = pm.Normal("z", 0.0, 1.0, dims=("subject", "param"))
        phi_raw = pm.Beta("phi_frac", 2.0, 2.0, dims="tissue")
        phi = pm.Deterministic("phi_deg", 90.0 * phi_raw, dims="tissue")

        logth = alpha[P.t_idx] + tau[None, :] * z[P.s_idx]
        if nC:
            logth = logth + pt.dot(pt.as_tensor_variable(P.X), beta)
        theta = pm.Deterministic("theta", pt.exp(logth), dims=("specimen", "param"))
        phi_j = (np.pi / 2) * phi_raw[P.t_idx]

        s_abs = pm.HalfNormal("s_abs", 5.0)
        s_rel = pm.HalfNormal("s_rel", 0.1)

        def gather(mask):
            j = df.loc[mask, "j"].to_numpy()
            col = lambda v: v[:, None]
            return (j, col(theta[j, 0]), col(theta[j, 1]), col(theta[j, 2]), col(theta[j, 3]),
                    col(phi_j[j]),
                    pt.as_tensor_variable(np.deg2rad(df.loc[mask, "load_angle_deg"].to_numpy())[:, None]),
                    pt.as_tensor_variable(df.loc[mask, "lam1"].to_numpy()[:, None]))

        # ---- biaxial block
        mb = (df.protocol == "biaxial").to_numpy()
        if mb.any():
            j, mu_, k1_, k2_, b_, ph_, an_, l1 = gather(mb)
            l2 = pt.as_tensor_variable(df.loc[mb, "lam2"].to_numpy()[:, None])
            s11, s22 = stress(l1, l2, mu_, k1_, k2_, density(b_, ph_, an_))
            for name, mod, obs in [("obs_bi_11", s11[:, 0], df.loc[mb, "sig11_kPa"]),
                                   ("obs_bi_22", s22[:, 0], df.loc[mb, "sig22_kPa"])]:
                pm.StudentT(name, nu=4, mu=mod, sigma=s_abs + s_rel * pt.abs(mod) + 1e-3,
                            observed=obs.to_numpy())
        # ---- uniaxial block: lateral stretch from s22 = 0
        mu_m = (df.protocol == "uniaxial").to_numpy()
        if mu_m.any():
            from pytensor.gradient import disconnected_grad as dg
            j, mu_, k1_, k2_, b_, ph_, an_, l1 = gather(mu_m)
            wrho = density(b_, ph_, an_)               # independent of lam2: compute once
            # (a) converge lam2 in NumPy (custom Op, no gradient graph) ...
            l2 = dg(LateralStretchOp(newton_iter)(l1, dg(mu_), dg(k1_), dg(k2_), dg(wrho)))
            # (b) ... then ONE connected Newton step: at convergence its derivative
            # equals the implicit-function-theorem sensitivity d lam2 / d theta exactly.
            _, f0, d = stress(l1, l2, mu_, k1_, k2_, wrho, deriv=True)
            l2 = l2 - f0 / (d + 1e-8)
            s11, _ = stress(l1, l2, mu_, k1_, k2_, wrho)
            s11 = s11[:, 0]
            pm.StudentT("obs_uni", nu=4, mu=s11, sigma=s_abs + s_rel * pt.abs(s11) + 1e-3,
                        observed=df.loc[mu_m, "sig11_kPa"].to_numpy())
    return model


# =============================================================================
# 5. Post-processing helpers
# =============================================================================
def model_curve(sample_row, th, phi_deg, lam_grid, lam2_grid=None):
    ang = np.deg2rad(sample_row.load_angle_deg)
    if sample_row.protocol == "uniaxial":
        return uniaxial_np(lam_grid, *th, np.deg2rad(phi_deg), ang)[0], None
    return ai_stress_np(lam_grid, lam2_grid, *th, np.deg2rad(phi_deg), ang)


def post_array(idata, var):
    return idata.posterior[var].stack(draw_all=("chain", "draw")).transpose("draw_all", ...).values


def noise_draws(idata, n):
    sa, sr = post_array(idata, "s_abs"), post_array(idata, "s_rel")
    idx = RNG.choice(len(sa), size=n, replace=True)
    return sa[idx], sr[idx]


def add_noise(curves, idata):
    """Posterior-predictive observations: model + Student-t(4) measurement noise."""
    sa, sr = noise_draws(idata, curves.shape[0])
    return curves + RNG.standard_t(4, curves.shape) * (sa[:, None] + sr[:, None] * np.abs(curves))


def population_draws(idata, P, tissue, x_new, n=300):
    """Posterior-predictive parameters for a NEW patient (covariates only)."""
    A = post_array(idata, "alpha")                  # (D, T, p)
    tau = post_array(idata, "tau")                  # (D, p)
    phi = post_array(idata, "phi_deg")              # (D, T)
    t = P.tissues.index(tissue)
    idx = RNG.choice(A.shape[0], size=min(n, A.shape[0]), replace=False)
    lt = A[idx, t, :] + tau[idx] * RNG.normal(size=(len(idx), len(PARAMS)))
    if P.covariates:
        B = post_array(idata, "beta")[idx]          # (n, C, p)
        xv = np.array([x_new.get(c, 0.0) for c in P.covariates])
        lt = lt + np.einsum("c,ncp->np", xv, B)
    return np.exp(lt), phi[idx, t]


# =============================================================================
# 6. Plotting
# =============================================================================
def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print("  saved", name)


def plot_data_overview(P, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = dict(zip(P.tissues, plt.cm.tab10.colors))
    for sid, g in P.df.groupby("sample_id"):
        t = g.tissue.iloc[0]
        axs[0].plot(g.lam1, g.sig11_kPa, ".-", color=cmap[t], alpha=0.35, ms=3, lw=0.8)
        if g.protocol.iloc[0] == "biaxial":
            axs[1].plot(g.lam2, g.sig22_kPa, ".-", color=cmap[t], alpha=0.35, ms=3, lw=0.8)
    for t in P.tissues:
        axs[0].plot([], [], color=cmap[t], label=t)
    axs[0].set(xlabel=r"$\lambda_1$", ylabel=r"$\sigma_{11}$ [kPa]", title="Loading-direction response")
    axs[1].set(xlabel=r"$\lambda_2$", ylabel=r"$\sigma_{22}$ [kPa]", title="Biaxial transverse response")
    axs[0].legend(fontsize=8)
    savefig(fig, out, "fig01_data_overview.png")

    s = P.samples.drop_duplicates("subject_id")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    if "age" in s:
        axs[0].hist(s.age.dropna(), bins=10, color="C0", ec="k"); axs[0].set(title="Age [yr]")
    if "sex" in s:
        s.sex.value_counts().plot.bar(ax=axs[1], color="C1", ec="k"); axs[1].set(title="Sex")
    if "smoker" in s:
        s.smoker.map({0: "non-smoker", 1: "smoker"}).value_counts().plot.bar(ax=axs[2], color="C2", ec="k")
        axs[2].set(title="Smoking history")
    savefig(fig, out, "fig02_covariates.png")


def _rhat_ess(x):
    """Split-R-hat and crude ESS for array (chain, draw)."""
    c, n = x.shape
    h = n // 2
    xs = np.concatenate([x[:, :h], x[:, h:2 * h]], 0)
    m = xs.shape[1]
    W = xs.var(1, ddof=1).mean(); B = m * xs.mean(1).var(ddof=1)
    rhat = np.sqrt(((m - 1) / m * W + B / m) / W) if W > 0 else np.nan
    return rhat


def plot_diagnostics(idata, P, out):
    """Version-independent trace plots, R-hat histogram and posterior summary."""
    names = ["tau", "phi_deg", "s_abs", "s_rel"] + (["beta"] if P.covariates else [])
    rows, panels = [], []
    for v in ["alpha"] + names:
        arr = np.asarray(idata.posterior[v].values)          # (chain, draw, ...)
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for k in range(flat.shape[2]):
            x = flat[:, :, k]
            lab = f"{v}[{k}]" if flat.shape[2] > 1 else v
            rows.append(dict(var=lab, mean=x.mean(), sd=x.std(), hdi_3=np.percentile(x, 3),
                             hdi_97=np.percentile(x, 97), r_hat=_rhat_ess(x)))
            if v != "alpha":
                panels.append((lab, x))
    summ = pd.DataFrame(rows).set_index("var")
    summ.round(4).to_csv(os.path.join(out, "posterior_summary.csv"))
    panels = panels[:12]
    fig, axs = plt.subplots(len(panels), 2, figsize=(11, 1.4 * len(panels)), squeeze=False)
    for i, (lab, x) in enumerate(panels):
        for c in range(x.shape[0]):
            axs[i, 0].hist(x[c], bins=30, histtype="step", density=True)
            axs[i, 1].plot(x[c], lw=0.5)
        axs[i, 0].set_ylabel(lab, fontsize=7, rotation=0, ha="right")
        axs[i, 0].set_yticks([]); axs[i, 0].tick_params(labelsize=7); axs[i, 1].tick_params(labelsize=7)
    axs[0, 0].set_title("marginal posterior (per chain)"); axs[0, 1].set_title("trace")
    savefig(fig, out, "fig03_trace.png")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(summ["r_hat"].dropna(), bins=20, color="C3", ec="k")
    ax.axvline(1.01, ls="--", c="k"); ax.set(title=r"split-$\hat R$ of hyper-parameters", xlabel=r"$\hat R$")
    savefig(fig, out, "fig04_rhat.png")
    return summ


def plot_beta_forest(idata, P, out, truth_beta=None):
    if not P.covariates:
        return
    B = post_array(idata, "beta")                   # (D, C, p)
    fig, axs = plt.subplots(1, len(PARAMS), figsize=(13, 3.4), sharey=True)
    for k, p in enumerate(PARAMS):
        ax = axs[k]
        for c, cov in enumerate(P.covariates):
            q = np.percentile(B[:, c, k], [3, 25, 50, 75, 97])
            ax.plot([q[0], q[4]], [c, c], c="C0", lw=1.2)
            ax.plot([q[1], q[3]], [c, c], c="C0", lw=4)
            ax.plot(q[2], c, "o", c="w", mec="C0")
            if truth_beta is not None:
                ax.plot(truth_beta[cov][p], c, "rx", ms=9, mew=2)
        ax.axvline(0, c="k", ls=":")
        ax.set(title=PARAM_TEX[p], xlabel=r"effect on $\log\theta$")
        ax.set_yticks(range(len(P.covariates)), P.covariates)
    fig.suptitle("Covariate effects (94% / 50% HDI-like intervals" +
                 ("; red x = ground truth)" if truth_beta else ")"))
    savefig(fig, out, "fig05_covariate_effects.png")

    # multiplicative effects table
    rows = []
    for c, cov in enumerate(P.covariates):
        for k, p in enumerate(PARAMS):
            v = np.exp(B[:, c, k])
            rows.append(dict(covariate=cov, param=p, factor_median=np.median(v),
                             factor_lo=np.percentile(v, 3), factor_hi=np.percentile(v, 97),
                             prob_increase=np.mean(B[:, c, k] > 0)))
    pd.DataFrame(rows).to_csv(os.path.join(out, "covariate_effect_factors.csv"), index=False)


def plot_prior_posterior(idata, P, out):
    A = post_array(idata, "alpha")
    fig, axs = plt.subplots(len(P.tissues), len(PARAMS), figsize=(13, 2.6 * len(P.tissues)), squeeze=False)
    for t, tis in enumerate(P.tissues):
        for k, p in enumerate(PARAMS):
            ax = axs[t, k]
            xs = np.linspace(PRIOR_LOGMEAN[p] - 3, PRIOR_LOGMEAN[p] + 3, 200)
            ax.plot(np.exp(xs), np.exp(-0.5 * (xs - PRIOR_LOGMEAN[p]) ** 2) / np.sqrt(2 * np.pi) /
                    np.exp(xs), "k--", lw=1, label="prior")
            ax.hist(np.exp(A[:, t, k]), bins=40, density=True, color="C0", alpha=0.6, label="posterior")
            ax.set_xscale("log")
            ax.set(title=f"{tis}: " + PARAM_TEX[p] + " (baseline)")
            if t == 0 and k == 0:
                ax.legend(fontsize=7)
    savefig(fig, out, "fig06_prior_vs_posterior.png")


def plot_pairs(idata, P, out):
    A = post_array(idata, "alpha")[:, 0, :]
    n = len(PARAMS)
    fig, axs = plt.subplots(n, n, figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if i == j:
                ax.hist(A[:, i], bins=30, color="C0")
            elif i > j:
                ax.plot(A[:, j], A[:, i], ".", ms=1, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"r={np.corrcoef(A[:, i], A[:, j])[0, 1]:.2f}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            if i == n - 1: ax.set_xlabel(r"$\log$" + PARAMS[j])
            if j == 0: ax.set_ylabel(r"$\log$" + PARAMS[i])
    fig.suptitle(f"Posterior correlations of baseline parameters ({P.tissues[0]})")
    savefig(fig, out, "fig07_pairs.png")


def plot_fits(idata, P, out, n_show=12, n_draw=60):
    TH = post_array(idata, "theta")                 # (D, J, p)
    PH = post_array(idata, "phi_deg")
    sel = P.samples.sample_id.tolist()[:: max(1, len(P.samples) // n_show)][:n_show]
    nc = 4; nr = int(np.ceil(len(sel) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 3 * nr), squeeze=False)
    draws = RNG.choice(TH.shape[0], size=min(n_draw, TH.shape[0]), replace=False)
    for a, sid in enumerate(sel):
        ax = axs.flat[a]
        g = P.df[P.df.sample_id == sid]
        row = g.iloc[0]; j = int(row.j); t = P.tissues.index(row.tissue)
        lg = np.linspace(1.0, g.lam1.max() * 1.02, 30)
        l2g = None
        if row.protocol == "biaxial":
            l2g = np.interp(lg, np.r_[1.0, g.lam1], np.r_[1.0, g.lam2])
        curves = []
        for d in draws:
            s11, s22 = model_curve(row, TH[d, j], PH[d, t], lg, l2g)
            curves.append(s11)
        curves = np.array(curves)
        ax.fill_between(lg, *np.percentile(curves, [5, 95], axis=0), color="C0", alpha=0.3, label="90% CI")
        ax.plot(lg, np.median(curves, 0), "C0")
        ax.plot(g.lam1, g.sig11_kPa, "ko", ms=3, label="data")
        ax.set_title(f"{sid}\n{row.tissue}, {row.protocol}", fontsize=8)
        ax.set(xlabel=r"$\lambda_1$", ylabel=r"$\sigma_{11}$ [kPa]")
    axs.flat[0].legend(fontsize=7)
    for b in axs.flat[len(sel):]:
        b.axis("off")
    savefig(fig, out, "fig08_sample_fits.png")


def plot_params_vs_covariates(idata, P, out):
    TH = post_array(idata, "theta")
    med = np.median(TH, 0)
    s = P.samples.copy()
    for k, p in enumerate(PARAMS):
        s[p] = med[:, k]
    if "age" not in s:
        return
    fig, axs = plt.subplots(len(P.tissues), len(PARAMS), figsize=(13, 2.8 * len(P.tissues)), squeeze=False)
    for t, tis in enumerate(P.tissues):
        g = s[s.tissue == tis]
        for k, p in enumerate(PARAMS):
            ax = axs[t, k]
            for smk, col in [(0, "C0"), (1, "C3")]:
                for sx, mk in [("F", "o"), ("M", "s")]:
                    h = g[(g.get("smoker", 0) == smk) & (g.get("sex", "F") == sx)]
                    ax.plot(h.age, h[p], mk, color=col, mfc="none" if sx == "F" else col, alpha=0.8,
                            label=f"{'smoker' if smk else 'non-smoker'}, {sx}")
            ax.set_yscale("log"); ax.set(title=f"{tis}: {PARAM_TEX[p]}", xlabel="age [yr]")
    axs[0, 0].legend(fontsize=6)
    savefig(fig, out, "fig09_params_vs_covariates.png")


def plot_population_calibration(idata, P, out):
    scen = [("30 y, F, non-smoker", dict(age=30, sex="F", smoker=0)),
            ("50 y, M, non-smoker", dict(age=50, sex="M", smoker=0)),
            ("70 y, M, smoker", dict(age=70, sex="M", smoker=1)),
            ("70 y, F, smoker", dict(age=70, sex="F", smoker=1))]
    rows = []
    fig, axs = plt.subplots(1, len(P.tissues), figsize=(4.4 * len(P.tissues), 3.8), squeeze=False)
    lg = np.linspace(1.0, 1.25, 30)
    for t, tis in enumerate(P.tissues):
        ax = axs[0, t]
        for c, (lab, sc) in enumerate(scen):
            x = dict(age_z=(sc["age"] - 55) / 15, smoker=sc["smoker"], male=int(sc["sex"] == "M"))
            th, ph = population_draws(idata, P, tis, x, n=150)
            curves = np.array([uniaxial_np(lg, *th[i], np.deg2rad(ph[i]), 0.0)[0] for i in range(len(th))])
            ax.fill_between(lg, *np.percentile(curves, [10, 90], 0), color=f"C{c}", alpha=0.18)
            ax.plot(lg, np.median(curves, 0), color=f"C{c}", label=lab)
            q = np.percentile(th, [5, 50, 95], 0)
            rows.append(dict(tissue=tis, scenario=lab, **{f"{p}_median": q[1, k] for k, p in enumerate(PARAMS)},
                             **{f"{p}_p05": q[0, k] for k, p in enumerate(PARAMS)},
                             **{f"{p}_p95": q[2, k] for k, p in enumerate(PARAMS)},
                             phi_deg_median=np.median(ph)))
        ax.set(title=f"{tis}: population-calibrated\n(uniaxial, 0 deg, 80% predictive band)",
               xlabel=r"$\lambda_1$", ylabel=r"$\sigma_{11}$ [kPa]")
        ax.set_ylim(0, None)
    axs[0, 0].legend(fontsize=7)
    savefig(fig, out, "fig10_population_calibration.png")
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibrated_parameters_by_scenario.csv"), index=False)


def plot_fibre_density(idata, P, out):
    A = post_array(idata, "alpha"); PH = post_array(idata, "phi_deg")
    th = np.linspace(0, 2 * np.pi, 361)
    fig, axs = plt.subplots(1, len(P.tissues), subplot_kw=dict(projection="polar"),
                            figsize=(4 * len(P.tissues), 4), squeeze=False)
    for t, tis in enumerate(P.tissues):
        ax = axs[0, t]
        idx = RNG.choice(A.shape[0], min(200, A.shape[0]), replace=False)
        dens = np.array([fibre_density(th, np.deg2rad(PH[i, t]), np.exp(A[i, t, 3])) for i in idx])
        ax.fill_between(th, *np.percentile(dens, [5, 95], 0), color="C2", alpha=0.3)
        ax.plot(th, np.median(dens, 0), "C2")
        ax.set_title(f"{tis}\n" + rf"$\phi$={np.median(PH[:, t]):.1f}$^\circ$", fontsize=9)
    fig.suptitle(r"Fibre orientation density $\rho(\theta)$ (0$^\circ$ = reference axis)")
    savefig(fig, out, "fig11_fibre_density.png")


def plot_ppc_residuals(idata, P, out):
    TH = np.median(post_array(idata, "theta"), 0); PH = np.median(post_array(idata, "phi_deg"), 0)
    res, pred = [], []
    for sid, g in P.df.groupby("sample_id"):
        row = g.iloc[0]; j = int(row.j); t = P.tissues.index(row.tissue)
        s11, _ = model_curve(row, TH[j], PH[t], g.lam1.to_numpy(),
                             g.lam2.to_numpy() if row.protocol == "biaxial" else None)
        res += list(g.sig11_kPa.to_numpy() - s11); pred += list(s11)
    res, pred = np.array(res), np.array(pred)
    from scipy import stats
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    axs[0].plot(pred, pred + res, ".", ms=3, alpha=0.5); m = max(pred.max(), (pred + res).max())
    axs[0].plot([0, m], [0, m], "k--"); axs[0].set(xlabel="predicted [kPa]", ylabel="observed [kPa]",
                                                   title=f"R$^2$={1 - res.var() / (pred + res).var():.3f}")
    axs[1].plot(pred, res, ".", ms=3, alpha=0.5); axs[1].axhline(0, c="k")
    axs[1].set(xlabel="predicted [kPa]", ylabel="residual [kPa]", title="Residuals")
    stats.probplot(res / (np.abs(pred) * 0.05 + 1), dist=stats.t(4), plot=axs[2])
    axs[2].set_title("Scaled residual Q-Q (Student-t, nu=4)")
    savefig(fig, out, "fig12_residuals.png")


def plot_holdout(idata, P, df_hold, out):
    if df_hold is None or df_hold.empty:
        return
    sids = df_hold.sample_id.unique()[:8]
    nc = 4; nr = int(np.ceil(len(sids) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 3.1 * nr), squeeze=False)
    cover = []
    for a, sid in enumerate(sids):
        g = df_hold[df_hold.sample_id == sid]; row = g.iloc[0]; ax = axs.flat[a]
        x = dict(age_z=(row.get("age", 55) - 55) / 15, smoker=row.get("smoker", 0),
                 male=int(str(row.get("sex", "F")).upper().startswith("M")))
        th, ph = population_draws(idata, P, row.tissue, x, n=120)
        lg = g.lam1.to_numpy()
        l2 = g.lam2.to_numpy() if row.protocol == "biaxial" else None
        cur = np.array([model_curve(row, th[i], ph[i], lg, l2)[0] for i in range(len(th))])
        lo, hi = np.percentile(add_noise(cur, idata), [5, 95], 0)
        plo, phi_ = np.percentile(cur, [5, 95], 0)
        cover.append(np.mean((g.sig11_kPa >= lo) & (g.sig11_kPa <= hi)))
        ax.fill_between(lg, lo, hi, color="C1", alpha=0.2, label="90% predictive (incl. noise)")
        ax.fill_between(lg, plo, phi_, color="C1", alpha=0.45, label="90% parameter band")
        ax.plot(lg, np.median(cur, 0), "C1"); ax.plot(lg, g.sig11_kPa, "ko", ms=3, label="unseen data")
        ax.set_title(f"held-out {sid}\n(age {row.get('age', np.nan):.0f}, {row.get('sex', '?')}, "
                     f"smoker={row.get('smoker', '?')})", fontsize=8)
    axs.flat[0].legend(fontsize=7)
    for b in axs.flat[len(sids):]:
        b.axis("off")
    fig.suptitle(f"Covariate-only prediction of unseen subjects (90% predictive coverage = {np.mean(cover):.2f})")
    savefig(fig, out, "fig13_holdout_prediction.png")


def plot_truth_recovery(idata, P, truth, out):
    if truth is None:
        return
    TH = post_array(idata, "theta")
    med, lo, hi = np.median(TH, 0), np.percentile(TH, 5, 0), np.percentile(TH, 95, 0)
    s = P.samples[["sample_id", "subject_id", "tissue"]].copy()
    s = s.merge(truth, on=["subject_id", "tissue"], how="left")
    fig, axs = plt.subplots(1, len(PARAMS), figsize=(13, 3.3))
    for k, p in enumerate(PARAMS):
        ax = axs[k]
        ax.errorbar(s[p], med[:, k], yerr=[med[:, k] - lo[:, k], hi[:, k] - med[:, k]], fmt="o", ms=3,
                    alpha=0.6)
        m = [min(s[p].min(), lo[:, k].min()), max(s[p].max(), hi[:, k].max())]
        ax.plot(m, m, "k--"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel="true", ylabel="posterior", title=PARAM_TEX[p])
    fig.suptitle("Synthetic-cohort verification: recovery of sample-level parameters")
    savefig(fig, out, "fig14_truth_recovery.png")


# =============================================================================
# 7. Main
# =============================================================================
class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *e): return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "mendeley", "csv"], default="synthetic")
    ap.add_argument("--csv", default=None); ap.add_argument("--meta", default=None)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--n_subjects", type=int, default=24)
    ap.add_argument("--holdout", type=int, default=3, help="subjects held out for validation")
    ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--method", choices=["nuts", "advi"], default="nuts")
    ap.add_argument("--advi_iter", type=int, default=10000)
    ap.add_argument("--load", default=None, help="re-plot from a saved posterior.nc (no refit)")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    truth = None
    if a.source == "synthetic":
        df, truth = make_synthetic(a.n_subjects)
    elif a.source == "mendeley":
        df = load_mendeley(a.raw_dir, a.meta)
        if df is None:
            sys.exit("No Mendeley curves found. Download the datasets manually:\n"
                     "  https://data.mendeley.com/datasets/yfj4wfszbw/4\n"
                     "  https://data.mendeley.com/datasets/x64srrc39p/4\n"
                     f"and unzip them into {a.raw_dir}/<dataset_id>/ ; then re-run.")
    else:
        df = load_long_csv(a.csv)
    df.to_csv(os.path.join(a.outdir, "data_used_long_format.csv"), index=False)

    # hold-out subjects for covariate-only validation
    subj = df.subject_id.unique()
    hold = RNG.choice(subj, size=min(a.holdout, max(0, len(subj) - 3)), replace=False) if a.holdout else []
    df_hold = df[df.subject_id.isin(hold)].copy()
    P = prepare(df[~df.subject_id.isin(hold)])
    plot_data_overview(P, a.outdir)

    import pymc as pm
    if a.load:
        import arviz as az
        idata = az.from_netcdf(a.load)
        a.method = "loaded"
    model = build_model(P) if not a.load else None
    with (model if model is not None else _nullctx()):
        if a.method == "loaded":
            pass
        elif a.method == "nuts":
            idata = pm.sample(draws=a.draws, tune=a.tune, chains=a.chains, cores=a.cores,
                              target_accept=0.9, random_seed=1, init="adapt_diag")
        else:
            try:                                # location differs across PyMC versions
                from pymc.variational.callbacks import CheckParametersConvergence
                cbs = [CheckParametersConvergence(tolerance=1e-3)]
            except Exception:
                cbs = []
            # MAP start + small Adam steps + gradient clipping keeps ADVI stable with the
            # stiff exponential fibre law
            start = pm.find_MAP(maxeval=4000, progressbar=False, seed=1)
            advi = pm.ADVI(start=start, random_seed=1)
            approx = advi.fit(a.advi_iter, obj_optimizer=pm.adam(learning_rate=0.005),
                              total_grad_norm_constraint=100.0, callbacks=cbs, progressbar=False)
            idata = approx.sample(a.draws, random_seed=1)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(approx.hist, lw=0.6); ax.set_yscale("symlog")
            ax.set(xlabel="iteration", ylabel="-ELBO", title="ADVI convergence")
            savefig(fig, a.outdir, "fig03_advi_elbo.png")
    try:
        if a.method != "loaded":
            idata.to_netcdf(os.path.join(a.outdir, "posterior.nc"))
    except Exception as exc:
        print(f"  could not write posterior.nc ({exc}); pip install h5netcdf")

    if a.method != "advi" and idata.posterior["tau"].shape[0] > 1:   # MCMC posterior
        plot_diagnostics(idata, P, a.outdir)
    plot_beta_forest(idata, P, a.outdir, TRUE_BETA if truth is not None else None)
    plot_prior_posterior(idata, P, a.outdir)
    plot_pairs(idata, P, a.outdir)
    plot_fits(idata, P, a.outdir)
    plot_params_vs_covariates(idata, P, a.outdir)
    plot_population_calibration(idata, P, a.outdir)
    plot_fibre_density(idata, P, a.outdir)
    plot_ppc_residuals(idata, P, a.outdir)
    plot_holdout(idata, P, df_hold, a.outdir)
    plot_truth_recovery(idata, P, truth, a.outdir)
    print(f"\nDone. All figures/tables in ./{a.outdir}/")


if __name__ == "__main__":
    main()
