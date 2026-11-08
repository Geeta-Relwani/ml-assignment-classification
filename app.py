"""Streamlit front end for the five trained purchase-intent classifiers.

The app never trains anything. It loads the Pipeline objects saved in model/ by
src/train_models.py and scores whatever test CSV is supplied, so the numbers it
shows are genuine held-out results rather than a model re-fitted on its own
evaluation data.

Run locally with:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import (  # noqa: E402  (import follows the sys.path setup above)
    METRIC_ORDER,
    compute_metrics,
    confusion,
    decision_scores,
    roc_points,
)
from preprocessing import FEATURES, TARGET  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "model"
BUNDLED_TEST_CSV = PROJECT_ROOT / "test_data.csv"

# Display name -> saved filename stem. Mirrors MODELS in src/train_models.py.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

PALETTE = {
    "ink": "#13203a",
    "accent": "#2f6f6b",
    "accent_soft": "#e4efed",
    "muted": "#6b7a90",
    "line": "#dde3ec",
}

st.set_page_config(
    page_title="Purchase Intent Classifier Lab",
    page_icon="📈",
    layout="wide",
)

# A little styling so the page does not read as an untouched Streamlit default:
# tighter metric cards, a muted rule under the header, calmer tab labels.
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    .lab-title {{
        font-size: 2.05rem; font-weight: 700; color: {PALETTE['ink']};
        letter-spacing: -0.02em; margin-bottom: 0.15rem;
    }}
    .lab-sub {{
        color: {PALETTE['muted']}; font-size: 0.97rem; margin-bottom: 1.1rem;
        padding-bottom: 1.1rem; border-bottom: 1px solid {PALETTE['line']};
    }}
    div[data-testid="stMetric"] {{
        background: {PALETTE['accent_soft']}; border-radius: 10px;
        padding: 0.7rem 0.75rem; border: 1px solid {PALETTE['line']};
    }}
    /* Six metric cards share one row, so the value has to stay small enough
       not to be truncated to an ellipsis in a narrow column. */
    div[data-testid="stMetricValue"] {{
        font-size: 1.1rem; color: {PALETTE['ink']};
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.8rem; color: {PALETTE['muted']};
    }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load every saved pipeline once per server process.

    cache_resource rather than cache_data: these are live sklearn objects, not
    serialisable values, and they should be shared across reruns rather than
    copied.

    joblib.load unpickles, which is only safe on trusted files. These are read
    from this repository's own model/ directory, written by
    src/train_models.py - never from user input. The CSV a visitor uploads is
    parsed by pandas and is never unpickled.
    """
    models = {}
    for name, stem in MODEL_FILES.items():
        path = MODEL_DIR / f"{stem}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def load_reference_metrics() -> dict:
    """Read the metrics recorded during training, if the file is present."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def validate(frame: pd.DataFrame) -> list:
    """Return the list of required columns missing from an uploaded frame."""
    required = FEATURES + [TARGET]
    return [column for column in required if column not in frame.columns]


def coerce_target(series: pd.Series) -> pd.Series:
    """Accept the target as True/False, "True"/"False" or 1/0 and return ints.

    Round-tripping a CSV through Excel is enough to turn booleans into strings,
    and an upload failing for that reason would be a poor experience.
    """
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .astype("Int64")
        )
    return series.astype(int)


def metrics_table(models: dict, X, y) -> pd.DataFrame:
    """Score every loaded model on the supplied data."""
    rows = {name: compute_metrics(y, model.predict(X), decision_scores(model, X))
            for name, model in models.items()}
    return pd.DataFrame(rows).T[METRIC_ORDER]


def plot_confusion(matrix, title: str):
    """Draw the confusion matrix as an annotated heatmap."""
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="BuGn",
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set_xlabel("Predicted", fontsize=9)
    axis.set_ylabel("Actual", fontsize=9)
    axis.set_title(title, fontsize=10, color=PALETTE["ink"])
    figure.tight_layout()
    return figure


def plot_roc(models: dict, X, y):
    """Overlay the ROC curve of every model on one set of axes."""
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    for name, model in models.items():
        fpr, tpr = roc_points(y, decision_scores(model, X))
        axis.plot(fpr, tpr, linewidth=1.8, label=name)
    # The diagonal is the no-skill reference: a model on this line is guessing.
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=PALETTE["muted"])
    axis.set_xlabel("False positive rate", fontsize=9)
    axis.set_ylabel("True positive rate", fontsize=9)
    axis.set_title("ROC curves on the supplied test data", fontsize=10)
    axis.legend(fontsize=8, loc="lower right", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


st.markdown(
    '<div class="lab-title">Purchase Intent Classifier Lab</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="lab-sub">Five pre-trained classifiers scored side by side on '
    "online shopping sessions from the UCI Online Shoppers Purchasing Intention "
    "dataset. Upload a test CSV and the saved models score it as they are &mdash; "
    "nothing is retrained here.</div>",
    unsafe_allow_html=True,
)

models = load_models()
if not models:
    st.error(
        "No saved models found in model/. Run `python src/train_models.py` "
        "to build them before starting the app."
    )
    st.stop()

# --- Sidebar: where the data comes from and which model to inspect ----------
with st.sidebar:
    st.header("Test data")
    # A radio rather than a checkbox plus uploader: a checkbox would keep its
    # earlier value after a file is dropped in, silently scoring the bundled
    # data instead of the upload.
    source_choice = st.radio(
        "Data source",
        ["Bundled test_data.csv", "Upload my own CSV"],
        help="The bundled file is the held-out 20% split that no model saw during training.",
    )
    uploaded = None
    if source_choice == "Upload my own CSV":
        uploaded = st.file_uploader(
            "Test CSV",
            type="csv",
            help="Must contain all 17 feature columns plus the Revenue target.",
        )

    st.divider()
    st.header("Model")
    selected_model = st.selectbox("Inspect a single model", list(models.keys()))

# Resolve the data source before anything is scored.
if source_choice == "Upload my own CSV":
    if uploaded is None:
        st.info("Choose a CSV in the sidebar to score it.")
        st.stop()
    raw = pd.read_csv(uploaded)
    source_label = uploaded.name
elif BUNDLED_TEST_CSV.exists():
    raw = pd.read_csv(BUNDLED_TEST_CSV)
    source_label = "test_data.csv"
else:
    st.error("test_data.csv is missing. Run `python src/export_test_data.py` to rebuild it.")
    st.stop()

missing = validate(raw)
if missing:
    st.error(
        "The CSV is missing required columns: " + ", ".join(missing) +
        ". It needs every feature column plus the target column `Revenue`."
    )
    st.stop()

X = raw[FEATURES]
y = coerce_target(raw[TARGET])
if y.isna().any():
    st.error(
        "The Revenue column contains values that are not True/False or 1/0."
    )
    st.stop()
y = y.astype(int)

summary = st.columns(4)
summary[0].metric("Rows scored", f"{len(raw):,}")
summary[1].metric("Features", len(FEATURES))
summary[2].metric("Purchases", f"{int(y.sum()):,}")
summary[3].metric("Purchase rate", f"{y.mean():.1%}")
st.caption(f"Scoring **{source_label}** with models loaded from `model/`.")

single_tab, compare_tab, roc_tab, reference_tab = st.tabs(
    ["Single model", "All five compared", "ROC curves", "Training reference"]
)

# --- Tab 1: one model in detail --------------------------------------------
with single_tab:
    model = models[selected_model]
    predictions = model.predict(X)
    scores = compute_metrics(y, predictions, decision_scores(model, X))

    st.subheader(selected_model)
    cards = st.columns(6)
    for card, metric in zip(cards, METRIC_ORDER):
        card.metric(metric, f"{scores[metric]:.4f}")

    left, right = st.columns([1, 1.25])
    with left:
        st.pyplot(plot_confusion(confusion(y, predictions), "Confusion matrix"))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["No purchase", "Purchase"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).T.round(4),
            use_container_width=True,
        )

# --- Tab 2: the comparison table the assignment asks for --------------------
with compare_tab:
    st.subheader("Metrics for all five models on this data")
    table = metrics_table(models, X, y)
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color=PALETTE["accent_soft"]),
        use_container_width=True,
    )
    st.caption("Best value in each column is highlighted.")

    melted = table.reset_index(names="Model").melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    figure, axis = plt.subplots(figsize=(9, 3.8))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.legend(fontsize=8, ncol=3, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.12))
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure)

    winner = table["F1"].idxmax()
    st.success(f"Highest F1 on this data: **{winner}** ({table.loc[winner, 'F1']:.4f}).")

# --- Tab 3: ROC curves ------------------------------------------------------
with roc_tab:
    st.subheader("Ranking quality across all thresholds")
    st.pyplot(plot_roc(models, X, y))
    st.caption(
        "AUC summarises these curves in one number. It reflects how well a model "
        "ranks purchasers above non-purchasers, independently of the 0.5 cut-off "
        "the other five metrics depend on."
    )

# --- Tab 4: what the training run recorded ----------------------------------
with reference_tab:
    reference = load_reference_metrics()
    if not reference:
        st.info("model/metrics.json was not found.")
    else:
        st.subheader("Recorded at training time")
        st.caption(
            f"{reference['dataset']} — {reference['n_train']:,} training rows, "
            f"{reference['n_test']:,} held-out rows, random_state="
            f"{reference['random_state']}."
        )
        st.dataframe(
            pd.DataFrame(reference["metrics"]).T[METRIC_ORDER].style.format("{:.4f}"),
            use_container_width=True,
        )
        st.markdown("**Hyperparameters chosen by cross-validation**")
        st.json(reference.get("best_params", {}))
