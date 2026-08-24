import base64
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from metrics import extract_metrics

HEALTHCARE_DIR = Path(__file__).resolve().parent.parent / "healthcare"

CHART_TAGS = {
    "ROC curve": r"roc_curve|roc_auc",
    "Confusion matrix": r"confusion_matrix",
    "Boxplot": r"\bboxplot\b",
    "Distribution": r"distplot|histplot|kdeplot|\.hist\(",
    "Correlation / heatmap": r"heatmap|\.corr\(",
    "Pairplot": r"pairplot",
    "Count plot": r"countplot",
    "Feature importance": r"feature_importances_|importance",
    "Scatter": r"scatter",
    "Bar chart": r"\bbarplot\b|\.bar\(",
}


def tag_source(source: str) -> list[str]:
    tags = [name for name, pattern in CHART_TAGS.items() if re.search(pattern, source, re.IGNORECASE)]
    return tags or ["Other"]


@st.cache_data(show_spinner=False)
def list_notebooks() -> list[Path]:
    return sorted(HEALTHCARE_DIR.glob("*.ipynb"))


@st.cache_data(show_spinner=False)
def load_notebook_images(nb_path_str: str) -> list[dict]:
    nb_path = Path(nb_path_str)
    nb = json.loads(nb_path.read_text())
    items = []
    preceding_markdown = ""
    for i, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown":
            preceding_markdown = source.strip()
            continue
        if cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if not png:
                continue
            items.append(
                {
                    "cell_index": i,
                    "source": source,
                    "markdown": preceding_markdown,
                    "png": png,
                    "tags": tag_source(source),
                }
            )
        preceding_markdown = ""
    return items


@st.cache_data(show_spinner=False)
def load_all_images(notebook_paths: tuple[str, ...]) -> dict[str, list[dict]]:
    return {p: load_notebook_images(p) for p in notebook_paths}


@st.cache_data(show_spinner=False)
def load_metrics_df(notebook_paths: tuple[str, ...]) -> pd.DataFrame:
    rows = extract_metrics([Path(p) for p in notebook_paths])
    return pd.DataFrame(rows)


CATEGORY_COLORS = {
    cat: color
    for cat, color in zip(
        [
            "RAC 2022 Ensemble",
            "SMOTE + Resampling Ensemble",
            "SMOTE",
            "No-SMOTE Baseline",
            "Undersampling",
            "Oversampling",
            "Cost-sensitive",
            "Feature-engineered (paper3)",
            "Feature engineering",
            "Imputation technique",
            "Heart-parameters model",
            "Metabolic-parameters model",
            "CVS/Respiratory model",
            "Sepsis prediction (full pipeline)",
            "Class-imbalance exploration",
            "XGBoost model",
            "EDA / Preprocessing",
            "Other",
        ],
        px.colors.qualitative.Bold + px.colors.qualitative.Pastel,
    )
}


def render_gallery(images: list[dict], cols_per_row: int, notebook_label: str | None = None):
    if not images:
        st.info("No plots match the current filters.")
        return
    for row_start in range(0, len(images), cols_per_row):
        row_items = images[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                st.image(base64.b64decode(item["png"]), width="stretch")
                tags_label = " / ".join(item["tags"])
                title = item.get("_nb_name", notebook_label)
                label = f"{title} — cell {item['cell_index']} — {tags_label}" if title else f"cell {item['cell_index']} — {tags_label}"
                with st.expander(label):
                    if item["markdown"]:
                        st.markdown(item["markdown"])
                    st.code(item["source"], language="python")


st.set_page_config(page_title="Sepsis ML Research Explorer", layout="wide")

notebooks = list_notebooks()
if not notebooks:
    st.warning(f"No notebooks found in {HEALTHCARE_DIR}")
    st.stop()

notebook_paths = tuple(str(p) for p in notebooks)
all_images = load_all_images(notebook_paths)
image_counts = {p: len(imgs) for p, imgs in all_images.items()}
total_images = sum(image_counts.values())

metrics_df = load_metrics_df(notebook_paths)

st.title("🩺 Sepsis Prediction: A Class-Imbalance Research Trail")
st.caption(
    "This repository is a running research log: predicting sepsis onset from ICU vital "
    "signs (heart rate, respiration, blood pressure, O₂ saturation, temperature) while "
    "fighting severe class imbalance — sepsis-positive readings are a small minority of "
    "the data. Every number below is pulled live from the notebooks' own saved outputs."
)

tab_story, tab_explorer, tab_gallery = st.tabs(["📖 The Story", "📊 Performance Explorer", "🖼️ Visual Gallery"])

# ---------------------------------------------------------------- Story tab
with tab_story:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notebooks analyzed", len(notebooks))
    c2.metric("Logged experiments", len(metrics_df))
    c3.metric("Saved charts extracted", total_images)
    if not metrics_df.empty:
        best = metrics_df.loc[metrics_df["value"].idxmax()]
        c4.metric("Best score seen", f"{best['value']:.0%}", help=f"{best['model']} — {best['notebook']} ({best['metric']})")

    st.markdown("### The arc")
    st.markdown(
        """
1. **Baseline models** are trained directly on the raw, heavily imbalanced ICU vitals.
2. **Imputation strategies** (mean/median, KNN, MICE, datawig) fill in missing vitals before modeling.
3. **Class-imbalance techniques** — SMOTE, oversampling, undersampling, cost-sensitive learning —
   are layered on top to stop models from just predicting "no sepsis" for everyone.
4. **Ensembles** (RAC 2022, bagging/boosting stacks) combine several of the above.

The chart below aggregates every ROC AUC score logged across the notebooks, grouped by
which of these techniques produced it — the closest thing this repo has to a scoreboard.
        """
    )

    roc_df = metrics_df[metrics_df["metric"].isin(["ROC AUC", "Mean ROC AUC"])]
    if not roc_df.empty:
        cat_summary = (
            roc_df.groupby("category")["value"]
            .agg(mean="mean", max="max", n="count")
            .sort_values("mean", ascending=False)
            .reset_index()
        )

        best_row = cat_summary.iloc[0]
        baseline_row = cat_summary[cat_summary["category"] == "No-SMOTE Baseline"]
        if not baseline_row.empty:
            delta = (best_row["mean"] - baseline_row.iloc[0]["mean"]) * 100
            st.info(
                f"**{best_row['category']}** leads with a mean ROC AUC of "
                f"**{best_row['mean']:.0%}** across {int(best_row['n'])} logged runs — "
                f"**{delta:.0f} points** above the No-SMOTE baseline "
                f"({baseline_row.iloc[0]['mean']:.0%})."
            )
        else:
            st.info(
                f"**{best_row['category']}** leads with a mean ROC AUC of "
                f"**{best_row['mean']:.0%}** across {int(best_row['n'])} logged runs."
            )

        fig = px.bar(
            cat_summary,
            x="mean",
            y="category",
            orientation="h",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            hover_data={"mean": ":.1%", "max": ":.1%", "n": True, "category": False},
            labels={"mean": "Mean ROC AUC", "category": ""},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, height=460)
        fig.update_traces(texttemplate="%{x:.0%}", textposition="outside")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig)
        st.caption("Bar = mean ROC AUC per technique family. Open **Performance Explorer** to filter by metric and see every individual run.")
    else:
        st.info("No ROC AUC scores were found in the saved notebook outputs.")

# ------------------------------------------------------------ Explorer tab
with tab_explorer:
    if metrics_df.empty:
        st.info("No metrics could be extracted from the notebook outputs.")
    else:
        st.markdown("Filter the full set of logged experiment results and compare techniques head-to-head.")
        f1, f2 = st.columns([1, 2])
        with f1:
            metric_choice = st.selectbox("Metric", sorted(metrics_df["metric"].unique()), index=0)
        with f2:
            categories = sorted(metrics_df["category"].unique())
            chosen_categories = st.multiselect("Technique / category", categories, default=categories)

        filtered = metrics_df[
            (metrics_df["metric"] == metric_choice) & (metrics_df["category"].isin(chosen_categories))
        ]

        if filtered.empty:
            st.info("No experiments match this filter.")
        else:
            fig = px.strip(
                filtered,
                x="category",
                y="value",
                color="category",
                color_discrete_map=CATEGORY_COLORS,
                hover_data={"notebook": True, "model": True, "value": ":.1%", "category": False},
                labels={"value": metric_choice, "category": ""},
            )
            box = px.box(filtered, x="category", y="value", color="category", color_discrete_map=CATEGORY_COLORS)
            for trace in box.data:
                trace.update(opacity=0.35, showlegend=False, boxpoints=False)
                fig.add_trace(trace)
            fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=520)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig)

            st.markdown(f"**Leaderboard — top runs by {metric_choice}**")
            leaderboard = (
                filtered.sort_values("value", ascending=False)
                .head(15)[["value", "model", "category", "notebook"]]
                .rename(columns={"value": metric_choice})
            )
            leaderboard[metric_choice] = leaderboard[metric_choice].map(lambda v: f"{v:.1%}")
            st.dataframe(leaderboard, hide_index=True)

# ------------------------------------------------------------- Gallery tab
with tab_gallery:
    st.markdown("Every plot already saved inside the notebooks' outputs — the evidence behind the numbers above.")
    g1, g2, g3 = st.columns([1.4, 1.4, 1])
    with g1:
        mode = st.radio("View", ["Browse by notebook", "Browse all"], horizontal=True)
    with g3:
        cols_per_row = st.slider("Images per row", 1, 4, 3)

    if mode == "Browse by notebook":
        options = [p for p in notebook_paths if image_counts[p] > 0]
        if not options:
            st.info("None of the notebooks have saved plot outputs.")
        else:
            with g2:
                selected = st.selectbox(
                    "Notebook",
                    options,
                    format_func=lambda p: f"{Path(p).name} ({image_counts[p]})",
                )
            images = all_images[selected]
            available_tags = sorted({t for item in images for t in item["tags"]})
            chosen_tags = st.multiselect("Filter by chart type", available_tags, key="nb_tags")
            if chosen_tags:
                images = [item for item in images if set(item["tags"]) & set(chosen_tags)]
            st.subheader(f"{Path(selected).name} — {len(images)} plot(s)")
            render_gallery(images, cols_per_row)
    else:
        all_tags = sorted({t for imgs in all_images.values() for item in imgs for t in item["tags"]})
        with g2:
            chosen_tags = st.multiselect("Filter by chart type", all_tags, key="all_tags")

        combined = []
        for p, imgs in all_images.items():
            for item in imgs:
                enriched = dict(item)
                enriched["_nb_name"] = Path(p).name
                combined.append(enriched)
        if chosen_tags:
            combined = [item for item in combined if set(item["tags"]) & set(chosen_tags)]

        st.subheader(f"All notebooks — {len(combined)} plot(s)")
        render_gallery(combined, cols_per_row)

    with st.expander("Notebook plot counts"):
        st.dataframe(
            pd.DataFrame(
                {
                    "notebook": [Path(p).name for p in notebook_paths],
                    "plots": [image_counts[p] for p in notebook_paths],
                }
            ),
            hide_index=True,
        )
