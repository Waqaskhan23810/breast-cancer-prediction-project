import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
)


@st.cache_data
def get_dataset():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = df["target"].astype(int)
    return df


@st.cache_resource
def train_pipeline():
    df = get_dataset()
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["malignant", "benign"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, cm, X, y


def render_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticklabels(["Malignant", "Benign"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    plt.tight_layout()
    return fig


st.title("🩺 Breast Cancer Prediction App")
st.markdown(
    "This app predicts whether a tumor is **benign** or **malignant** using the Breast Cancer Wisconsin dataset."
)

model, accuracy, report, cm, X, y = train_pipeline()
feature_names = list(X.columns)

df = get_dataset()

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Dataset Size", len(df))
col3.metric("Features", len(feature_names))

st.subheader("Model Performance")
perf_df = pd.DataFrame(
    {
        "Class": ["malignant", "benign", "accuracy", "macro avg", "weighted avg"],
        "Precision": [
            report["malignant"]["precision"],
            report["benign"]["precision"],
            report["accuracy"],
            report["macro avg"]["precision"],
            report["weighted avg"]["precision"],
        ],
        "Recall": [
            report["malignant"]["recall"],
            report["benign"]["recall"],
            report["accuracy"],
            report["macro avg"]["recall"],
            report["weighted avg"]["recall"],
        ],
        "F1-score": [
            report["malignant"]["f1-score"],
            report["benign"]["f1-score"],
            report["accuracy"],
            report["macro avg"]["f1-score"],
            report["weighted avg"]["f1-score"],
        ],
    }
)

st.dataframe(perf_df, width="stretch")
st.pyplot(render_confusion_matrix(cm))

st.subheader("Predict for a New Patient")
st.caption("Use the dataset average or enter your own tumor measurements for a custom diagnosis prediction.")

with st.form("prediction_form", clear_on_submit=False):
    prediction_mode = st.selectbox(
        "Choose input method",
        ["Use dataset average values", "Enter custom values manually"],
        index=1,
    )

    custom_values = {}
    for name in feature_names:
        col_values = X[name]
        default_value = float(col_values.mean())
        min_value = float(col_values.min())
        max_value = max(float(col_values.max()) * 1.2, float(col_values.max()) + 10.0)

        if prediction_mode == "Use dataset average values":
            current_value = default_value
            min_value = 0.0
        else:
            current_value = default_value
            min_value = 0.0

        custom_values[name] = st.number_input(
            f"{name}",
            min_value=min_value,
            max_value=max_value,
            value=current_value,
            step=0.01,
            format="%.2f",
            key=f"input_{name}",
        )

    submitted = st.form_submit_button("Predict Tumor Type")

if submitted:
    input_df = pd.DataFrame([custom_values], columns=feature_names)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    label = "Benign" if prediction == 1 else "Malignant"
    confidence = max(probabilities)

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    if prediction == 1:
        st.success("The tumor is likely benign.")
    else:
        st.error("The tumor is likely malignant.")

    st.write("Probability breakdown:")
    prob_df = pd.DataFrame(
        {
            "Class": ["Malignant", "Benign"],
            "Probability": probabilities,
        }
    )
    st.dataframe(prob_df, width="stretch")

st.subheader("Dataset Preview")
st.dataframe(df.head(), width="stretch")
