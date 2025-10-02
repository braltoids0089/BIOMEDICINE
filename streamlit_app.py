"""Streamlit app for Digital Twin Workflow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core_logic import (
    SIGS,
    compute_groups,
    example_drug_panel,
    load_expression_matrix,
    pathway_scores,
    plot_coheatmap,
    plot_frequency,
    plot_heatmap,
    plot_pca,
    plot_size_hist,
    run_patient_selection,
    run_pca,
    selection_statistics,
    synthetic_classification,
)

STYLES = """
<style>
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}
</style>
"""


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Digital Twin Therapy Explorer", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)
    st.title("Digital Twin Workflow: Gene Expression → Pathway-Guided Therapy")
    st.write(
        "This interactive app reproduces the end-to-end workflow from the notebook,"
        " starting from a TCGA Toil expression matrix and culminating in pathway-guided"
        " therapy selection using a QUBO formulation."
    )

    with st.sidebar:
        st.header("Configuration")
        use_demo = st.toggle("Use built-in demo cohort (no download)", value=True)
        max_samples = 24 if use_demo else 120
        default_samples = min(40, max_samples)
        n_samples = st.slider(
            "# of TCGA samples", min_value=8, max_value=max_samples, value=default_samples, step=8
        )
        lam = st.slider("λ (QUBO regularization)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
        base_overlap = st.slider(
            "Penalty per shared pathway", min_value=0.0, max_value=1.0, value=0.25, step=0.05
        )
        sparsity = st.slider("Sparsity penalty", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
        run_optimization = st.checkbox("Run therapy optimization", value=True)
        run_validation = st.checkbox("Run synthetic validation", value=False)

    st.subheader("1. Load expression matrix")
    expr = load_expression_matrix(n_samples, use_demo=use_demo)
    dataset_label = "demo cohort" if use_demo else "TCGA Toil subset"
    st.success(
        f"Expression matrix ready ({dataset_label}): {expr.shape[0]} genes × {expr.shape[1]} samples"
    )
    st.dataframe(expr.head())

    st.subheader("2. Principal component analysis")
    pc_df, explained = run_pca(expr)
    groups = compute_groups(pc_df.index)
    fig_pca = plot_pca(pc_df, explained, groups)
    st.pyplot(fig_pca)
    st.caption(
        f"Top PCs explain {explained[:3].sum()*100:.1f}% variance. Tumor/normal groups derived from TCGA barcodes."
    )

    st.subheader("3. Pathway scoring")
    P = pathway_scores(expr, SIGS)
    st.write("Pathway matrix shape:", P.shape)
    st.dataframe(P.head())
    st.pyplot(plot_heatmap(P))

    if run_optimization:
        st.subheader("4. Therapy selection via QUBO")
        selections, benefit_df, pathway_df = run_patient_selection(
            P, lam=lam, base_overlap=base_overlap, sparsity=sparsity
        )
        panel = example_drug_panel()
        drugs = list(panel.keys())
        freq_df, size_df, co_pct = selection_statistics(selections, drugs)

        st.markdown("**Selection frequency**")
        st.dataframe(freq_df)
        st.pyplot(plot_frequency(freq_df))

        st.markdown("**Selection size distribution**")
        st.dataframe(size_df)
        st.pyplot(plot_size_hist(size_df))

        st.markdown("**Drug co-selection**")
        st.dataframe(co_pct.round(1))
        st.pyplot(plot_coheatmap(co_pct))

        patient_table = pd.DataFrame(
            {
                "patient": P.columns,
                "selected_drugs": [", ".join(sel) if sel else "(none)" for sel in selections],
            }
        ).set_index("patient")
        st.markdown("**Patient-level selections**")
        st.dataframe(patient_table)

        st.markdown("**Drug benefit priors**")
        st.dataframe(benefit_df.round(3))

        st.markdown("**Top pathway context (per patient)**")
        st.dataframe(pathway_df.round(3))

        if run_validation:
            st.subheader("5. Synthetic classification validation")
            coef_df, roc_auc, pr_auc = synthetic_classification(selections, P)
            if coef_df.empty:
                st.warning("Synthetic labels collapsed to one class; increase sample count.")
            else:
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
                st.metric("PR-AUC", f"{pr_auc:.3f}")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(coef_df["drug"], coef_df["weight"], color="steelblue")
                ax.set_xlabel("Weight")
                ax.set_title("Logistic regression coefficients")
                fig.tight_layout()
                st.pyplot(fig)
                st.dataframe(coef_df)

    st.info(
        "All computations are performed on-demand with caching."
        " Increase the sample count for richer cohorts, or adjust penalties"
        " to explore alternative therapy panels."
    )


if __name__ == "__main__":
    main()