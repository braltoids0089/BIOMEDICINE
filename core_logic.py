"""Core logic for the Digital Twin Workflow."""
from __future__ import annotations

import gzip
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

TPM_URL = "https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz"
DATA_DIR = Path(".cache")
DATA_DIR.mkdir(exist_ok=True)

# Compact signature panel (from the notebook)
SIGS: Dict[str, List[str]] = {
    "REACTOME_SIGNALING_BY_EGFR": [
        "EGFR", "ERBB2", "ERBB3", "GRB2", "SOS1", "SHC1", "PTPN11", "KRAS", "NRAS",
        "HRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "PLCG1", "PIK3CA",
        "PIK3R1", "AKT1", "AKT2", "AKT3", "GAB1",
    ],
    "REACTOME_SIGNALING_BY_ALK": [
        "ALK", "EML4", "GRB2", "SHC1", "PIK3CA", "PIK3R1", "AKT1", "AKT2", "AKT3",
        "STAT3", "MAP2K1", "MAPK1", "MAPK3",
    ],
    "REACTOME_MAPK1_MAPK3_SIGNALING": [
        "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "DUSP6", "DUSP4", "FOS",
        "JUN", "EGFR",
    ],
    "REACTOME_PI3K_AKT_SIGNALING": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3",
        "PTEN", "MTOR", "RHEB",
    ],
    "REACTOME_MTORC1_MEDIATED_SIGNALLING": [
        "MTOR", "RPTOR", "MLST8", "RHEB", "TSC1", "TSC2", "EIF4EBP1", "RPS6KB1", "RPS6",
    ],
    "REACTOME_PD1_SIGNALING": [
        "PDCD1", "CD274", "PDCD1LG2", "JAK1", "JAK2", "STAT1", "IFNG", "GZMB", "LAG3",
        "TIGIT", "CXCL9", "CXCL10",
    ],
    "REACTOME_VEGFA_VEGFR2_SIGNALING_PATHWAY": [
        "VEGFA", "KDR", "FLT1", "FLT4", "PTPRB", "PLCG1", "MAP2K1", "MAPK1", "NOS3",
    ],
    "REACTOME_SIGNALING_BY_FGFR": [
        "FGFR1", "FGFR2", "FGFR3", "FGFR4", "FRS2", "PLCG1", "PIK3CA", "PIK3R1",
        "MAP2K1", "MAPK1",
    ],
}

SYM2ENSG: Dict[str, str] = {
    "EGFR": "ENSG00000146648", "ERBB2": "ENSG00000141736", "ERBB3": "ENSG00000065361",
    "GRB2": "ENSG00000177885", "SOS1": "ENSG00000115904", "SHC1": "ENSG00000154639",
    "PTPN11": "ENSG00000179295", "KRAS": "ENSG00000133703", "NRAS": "ENSG00000213281",
    "HRAS": "ENSG00000174775", "BRAF": "ENSG00000157764", "MAP2K1": "ENSG00000169032",
    "MAP2K2": "ENSG00000126934", "MAPK1": "ENSG00000100030", "MAPK3": "ENSG00000102882",
    "PLCG1": "ENSG00000124181", "PIK3CA": "ENSG00000121879", "PIK3R1": "ENSG00000145675",
    "AKT1": "ENSG00000142208", "AKT2": "ENSG00000105221", "AKT3": "ENSG00000117020",
    "GAB1": "ENSG00000117676", "ALK": "ENSG00000171094", "EML4": "ENSG00000143924",
    "STAT3": "ENSG00000168610", "DUSP6": "ENSG00000139318", "DUSP4": "ENSG00000120875",
    "FOS": "ENSG00000170345", "JUN": "ENSG00000177606", "RAF1": "ENSG00000132155",
    "PIK3CB": "ENSG00000119402", "PIK3CD": "ENSG00000171608", "PIK3R2": "ENSG00000189403",
    "PTEN": "ENSG00000171862", "MTOR": "ENSG00000198793", "RHEB": "ENSG00000106615",
    "RPTOR": "ENSG00000141564", "MLST8": "ENSG00000105705", "TSC1": "ENSG00000165699",
    "TSC2": "ENSG00000103197", "EIF4EBP1": "ENSG00000187840", "RPS6KB1": "ENSG00000108443",
    "RPS6": "ENSG00000137154", "PDCD1": "ENSG00000276977", "CD274": "ENSG00000120217",
    "PDCD1LG2": "ENSG00000197646", "JAK1": "ENSG00000162434", "JAK2": "ENSG00000096968",
    "STAT1": "ENSG00000115415", "IFNG": "ENSG00000111537", "GZMB": "ENSG00000100453",
    "LAG3": "ENSG00000089692", "TIGIT": "ENSG00000181847", "CXCL9": "ENSG00000138755",
    "CXCL10": "ENSG00000169245", "VEGFA": "ENSG00000112715", "KDR": "ENSG00000128052",
    "FLT1": "ENSG00000102755", "FLT4": "ENSG00000037280", "PTPRB": "ENSG00000160593",
    "NOS3": "ENSG00000164867", "FGFR1": "ENSG00000077782", "FGFR2": "ENSG00000066468",
    "FGFR3": "ENSG00000068078", "FGFR4": "ENSG00000069535", "FRS2": "ENSG00000181873",
}

def download_tpm() -> Path:
    path = DATA_DIR / "TcgaTargetGtex_rsem_gene_tpm.gz"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with requests.get(TPM_URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    return path

def detect_row_mode(path: Path, scan_rows: int = 50000) -> str:
    seen = Counter()
    total = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = pd.read_csv(fh, sep="\t", chunksize=200_000, usecols=[0], dtype=str, header=0)
        for chunk in reader:
            vals = chunk.iloc[:, 0].astype(str)
            head = vals.head(max(0, scan_rows - total))
            if head.empty:
                continue
            seen.update("ENSG" if v.startswith("ENSG") else "OTHER" for v in head)
            total += len(head)
            if total >= scan_rows:
                break
    ratio = seen["ENSG"] / max(1, (seen["ENSG"] + seen["OTHER"]))
    return "ensembl" if ratio >= 0.6 else "symbol"

def _normalize_row_id(value: str, mode: str) -> str:
    if mode == "ensembl":
        return value.split(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "", value)

def load_expression_matrix(n_samples: int, use_demo: bool) -> pd.DataFrame:
    if use_demo:
        path = Path("demo_data/expression_demo.csv")
        df = pd.read_csv(path, index_col=0)
        cols = list(df.columns)[:n_samples]
        expr = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
        gene_means = expr.mean(axis=1)
        expr = expr.apply(lambda col: col.fillna(gene_means), axis=0)
        return expr

    path = download_tpm()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n")
    cols = header.split("\t")
    gene_col = cols[0]
    sample_cols = [c for c in cols[1:] if isinstance(c, str) and c.startswith("TCGA-")]
    if not sample_cols:
        sample_cols = cols[1:]
    selected_samples = sample_cols[:n_samples]

    row_mode = detect_row_mode(path)
    if row_mode == "ensembl":
        target_rows = {SYM2ENSG[g] for g in SIGS for g in SIGS[g] if g in SYM2ENSG}
        id_map = {v: k for k, v in SYM2ENSG.items()}
    else:
        target_rows = {g for genes in SIGS.values() for g in genes}
        id_map = {}

    usecols = [gene_col] + selected_samples
    kept_frames = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = pd.read_csv(
            fh,
            sep="\t",
            chunksize=50_000,
            dtype=str,
            usecols=lambda c: (c == gene_col) or (c in usecols),
        )
        for chunk in reader:
            chunk = chunk.rename(columns={gene_col: "row_id"})
            ids = chunk["row_id"].astype(str).map(lambda v: _normalize_row_id(v, row_mode))
            mask = ids.isin(target_rows)
            if mask.any():
                sub = chunk.loc[mask].copy()
                sub["row_id"] = ids[mask]
                kept_frames.append(sub)
    if not kept_frames:
        raise RuntimeError("No gene signature rows matched the requested samples.")

    expr = (
        pd.concat(kept_frames, axis=0, ignore_index=False)
        .drop_duplicates(subset=["row_id"])
        .set_index("row_id")
    )
    if row_mode == "ensembl":
        expr.index = [id_map.get(idx, idx) for idx in expr.index]
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.loc[~expr.isna().all(axis=1)]
    expr = expr.loc[:, ~expr.isna().all(axis=0)]
    gene_means = expr.mean(axis=1)
    expr = expr.apply(lambda col: col.fillna(gene_means), axis=0)
    return expr

def compute_groups(sample_ids: Sequence[str]) -> pd.Series:
    def bucket(sample: str) -> str:
        parts = sample.split("-")
        if len(parts) >= 4 and parts[3][:2].isdigit():
            code = int(parts[3][:2])
        else:
            return "Other"
        if 1 <= code <= 9:
            return "Tumor"
        if 10 <= code <= 19:
            return "Normal"
        return "Other"

    return pd.Series({sid: bucket(sid) for sid in sample_ids})

def run_pca(expr: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = expr.T.copy()
    std = X.std(axis=0)
    X = X.loc[:, (std > 0).values]
    mu = X.mean(axis=0).to_numpy()
    sd = X.std(axis=0).to_numpy() + 1e-8
    Xz = (X.to_numpy() - mu) / sd
    U, S, _ = np.linalg.svd(Xz, full_matrices=False)
    scores = U * S
    explained = (S ** 2) / np.sum(S ** 2)
    pc_df = pd.DataFrame(scores[:, :3], index=X.index, columns=["PC1", "PC2", "PC3"])
    return pc_df, explained

def zscore_by_gene(expr: pd.DataFrame) -> pd.DataFrame:
    E = expr.copy()
    E = E.apply(pd.to_numeric, errors="coerce")
    E = E.loc[~E.isna().all(axis=1)]
    gene_means = E.mean(axis=1)
    E = E.apply(lambda col: col.fillna(gene_means), axis=0)
    mu = E.mean(axis=1)
    sd = E.std(axis=1) + 1e-8
    return (E.sub(mu, axis=0)).div(sd, axis=0)

def pathway_scores(expr: pd.DataFrame, signatures: Dict[str, Sequence[str]]) -> pd.DataFrame:
    Z = zscore_by_gene(expr)
    rows = []
    for pathway, genes in signatures.items():
        present = [g for g in genes if g in Z.index]
        if present:
            series = Z.loc[present].mean(axis=0)
        else:
            series = pd.Series([np.nan] * Z.shape[1], index=Z.columns)
        series.name = pathway
        rows.append(series)
    return pd.DataFrame(rows)

def example_drug_panel() -> Dict[str, List[str]]:
    return {
        "EGFRi": ["REACTOME_SIGNALING_BY_EGFR"], "ALKi": ["REACTOME_SIGNALING_BY_ALK"],
        "MEKi": ["REACTOME_MAPK1_MAPK3_SIGNALING"], "PI3Ki": ["REACTOME_PI3K_AKT_SIGNALING"],
        "mTORi": ["REACTOME_MTORC1_MEDIATED_SIGNALLING"], "PD1i": ["REACTOME_PD1_SIGNALING"],
        "VEGFi": ["REACTOME_VEGFA_VEGFR2_SIGNALING_PATHWAY"], "FGFRi": ["REACTOME_SIGNALING_BY_FGFR"],
    }

def patient_vector(P: pd.DataFrame, sample_id: str) -> pd.Series:
    z = (P - P.mean(axis=1).values.reshape(-1, 1)) / (P.std(axis=1).values.reshape(-1, 1) + 1e-8)
    return z[sample_id].fillna(0.0)

def drug_benefit_prior(z_path: pd.Series, panel: Dict[str, Sequence[str]]) -> pd.Series:
    scores = {
        drug: float(np.sum([max(z_path.get(p, 0.0), 0.0) for p in pathways]))
        for drug, pathways in panel.items()
    }
    series = pd.Series(scores)
    if series.max() > 0:
        series = series / series.max()
    return series

def build_penalty_matrix(
    drugs: Sequence[str], panel: Dict[str, Sequence[str]], base_overlap: float, sparsity: float
) -> np.ndarray:
    K = len(drugs)
    R = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            overlap = len(set(panel[drugs[i]]) & set(panel[drugs[j]]))
            if overlap:
                R[i, j] = R[j, i] = base_overlap * overlap
    for i in range(K):
        R[i, i] += sparsity
    return R

def exact_qubo_solve(b_hat: np.ndarray, R: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    K = len(b_hat)
    Q = lam * R.copy()
    q = -b_hat.copy() + np.diag(Q)
    np.fill_diagonal(Q, 0.0)
    best_energy = float("inf")
    best_x: np.ndarray | None = None
    upper_idx = np.triu_indices(K, 1)
    for mask in range(1 << K):
        x = np.fromiter(((mask >> i) & 1 for i in range(K)), dtype=np.int8)
        energy = float(np.dot(q, x) + 2.0 * np.sum(Q[upper_idx] * (x[upper_idx[0]] * x[upper_idx[1]])))
        if energy < best_energy:
            best_energy, best_x = energy, x
    assert best_x is not None
    return best_x, best_energy

def run_patient_selection(
    P: pd.DataFrame, lam: float, base_overlap: float, sparsity: float
) -> Tuple[List[List[str]], pd.DataFrame, pd.DataFrame]:
    panel = example_drug_panel()
    drugs = list(panel.keys())
    selections: List[List[str]] = []
    benefit_rows = []
    pathway_rows = []

    for sample_id in P.columns:
        z_path = patient_vector(P, sample_id)
        benefit = drug_benefit_prior(z_path, panel).reindex(drugs).fillna(0.0)
        R = build_penalty_matrix(drugs, panel, base_overlap=base_overlap, sparsity=sparsity)
        bits, _ = exact_qubo_solve(benefit.to_numpy(float), R, lam=lam)
        selected = [drugs[i] for i, bit in enumerate(bits) if bit == 1]
        selections.append(selected)
        benefit_rows.append({"patient": sample_id, **{drug: benefit.get(drug, 0.0) for drug in drugs}})
        top_pathways = z_path.abs().sort_values(ascending=False).head(5)
        pathway_rows.append({
            "patient": sample_id,
            **{pw: z_path[pw] for pw in top_pathways.index},
        })

    benefit_df = pd.DataFrame(benefit_rows).set_index("patient")
    pathway_df = pd.DataFrame(pathway_rows).set_index("patient")
    return selections, benefit_df, pathway_df

def selection_statistics(
    selections: Sequence[Sequence[str]], drugs: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flat = [drug for sel in selections for drug in sel]
    freq = Counter(flat)
    total = max(1, len(selections))
    freq_df = pd.DataFrame({"drug": drugs, "count": [freq.get(d, 0) for d in drugs]})
    freq_df["pct_patients"] = 100.0 * freq_df["count"] / total
    freq_df = freq_df.sort_values("pct_patients", ascending=False, ignore_index=True)

    sizes = pd.Series([len(set(sel)) for sel in selections])
    size_df = sizes.value_counts().sort_index().rename_axis("n_drugs").reset_index(name="n_patients")

    co_mat = pd.DataFrame(0, index=drugs, columns=drugs, dtype=float)
    for sel in selections:
        uniq = sorted(set(sel))
        for i, di in enumerate(uniq):
            co_mat.loc[di, di] += 1
            for dj in uniq[i + 1 :]:
                co_mat.loc[di, dj] += 1
                co_mat.loc[dj, di] += 1
    co_pct = co_mat / total * 100.0
    return freq_df, size_df, co_pct

def synthetic_classification(
    selections: Sequence[Sequence[str]], P: pd.DataFrame
) -> Tuple[pd.DataFrame, float, float]:
    panel = example_drug_panel()
    drugs = list(panel.keys())
    N = len(selections)
    X = np.zeros((N, len(drugs)))
    for i, sel in enumerate(selections):
        for drug in sel:
            if drug in panel:
                X[i, drugs.index(drug)] = 1

    risk_scores = []
    for sample_id in P.columns[:N]:
        z_path = patient_vector(P, sample_id)
        risk_scores.append(z_path.mean())
    risk_scores = np.asarray(risk_scores)
    y = (risk_scores > np.median(risk_scores)).astype(int)

    if len(np.unique(y)) < 2:
        return pd.DataFrame(), float("nan"), float("nan")

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_pred)
    pr_auc = average_precision_score(y, y_pred)

    coef_series = pd.Series(clf.coef_[0], index=drugs, name="weight").sort_values()
    coef_df = coef_series.reset_index().rename(columns={"index": "drug"})
    return coef_df, roc_auc, pr_auc

def plot_pca(pc_df: pd.DataFrame, explained: np.ndarray, groups: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = groups.reindex(pc_df.index).fillna("Other")
    for group in labels.unique():
        mask = labels == group
        ax.scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"], label=group, alpha=0.75)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title("PCA of expression signatures")
    ax.legend()
    return fig

def plot_heatmap(P: pd.DataFrame) -> plt.Figure:
    var = P.var(axis=1).sort_values(ascending=False)
    keep = list(var.index[: min(12, len(var))])
    Ps = P.loc[keep]
    Ps_vis = (Ps - Ps.mean(axis=1).values.reshape(-1, 1)) / (Ps.std(axis=1).values.reshape(-1, 1) + 1e-8)
    fig, ax = plt.subplots(figsize=(min(12, 0.3 * Ps_vis.shape[1] + 3), 6))
    im = ax.imshow(Ps_vis.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(Ps_vis.shape[1]))
    ax.set_xticklabels(Ps_vis.columns.str.slice(0, 12), rotation=90)
    ax.set_yticks(range(Ps_vis.shape[0]))
    ax.set_yticklabels(Ps_vis.index)
    ax.set_title("Pathway activity (variance-ranked subset)")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    return fig

def plot_frequency(freq_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(freq_df))
    ax.bar(positions, freq_df["pct_patients"], color="steelblue")
    ax.set_ylabel("% patients")
    ax.set_title("Drug selection frequency")
    ax.set_xticks(positions)
    ax.set_xticklabels(freq_df["drug"], rotation=45, ha="right")
    fig.tight_layout()
    return fig

def plot_size_hist(size_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(size_df["n_drugs"], size_df["n_patients"], color="mediumpurple")
    ax.set_xlabel("# drugs selected")
    ax.set_ylabel("# patients")
    ax.set_title("Selection size distribution")
    fig.tight_layout()
    return fig

def plot_coheatmap(co_pct: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(co_pct.values, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(co_pct.columns)))
    ax.set_xticklabels(co_pct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(co_pct.index)))
    ax.set_yticklabels(co_pct.index)
    ax.set_title("Drug co-selection (% patients)")
    fig.colorbar(im, ax=ax, label="% patients")
    fig.tight_layout()
    return fig