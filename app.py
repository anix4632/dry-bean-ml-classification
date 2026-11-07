
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Streamlit page configuration

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write(
    "Machine Learning Model Comparison and Evaluation "
    "using the UCI Dry Bean Dataset."
)

st.markdown("---")

# 2. Model file configuration

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_nb.joblib",
    "SVM": "model/svm.joblib"
}

# 3. Load saved models

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_FILES.items():

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:
    trained_models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

# 4. Upload test dataset

st.header("1. Upload Test Dataset")

st.write(
    "Upload a CSV file containing all 16 input features "
    "and the actual Class column."
)

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)


# 5. Select machine learning model

st.header("2. Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(MODEL_FILES.keys())
)

selected_model = trained_models[selected_model_name]

st.info(f"Selected model: {selected_model_name}")

# 6. Evaluate uploaded dataset

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(uploaded_file)

        st.header("3. Dataset Information")

        st.write("Uploaded dataset shape:", test_df.shape)

        st.dataframe(test_df.head())

        # Check target column
        if "Class" not in test_df.columns:
            st.error(
                "The uploaded CSV must contain the target column 'Class'."
            )
            st.stop()

        # Get input feature names from trained model
        required_features = list(selected_model.feature_names_in_)

        missing_features = [
            col for col in required_features
            if col not in test_df.columns
        ]

        if missing_features:
            st.error(
                f"Missing required input features: {missing_features}"
            )
            st.stop()
        # Prepare input features and target values
        X_test_upload = test_df[required_features].copy()

        if test_df["Class"].isnull().any():
            st.error("The target column 'Class' contains missing values.")
            st.stop()

        y_test_upload = test_df["Class"].astype(str)

        # Validate numeric input features
        try:
            X_test_upload = X_test_upload.apply(
            pd.to_numeric, errors="raise"
            )
        except (ValueError, TypeError):
            st.error("All input features must contain numeric values.")
            st.stop()

        # Check for missing input values
        if X_test_upload.isnull().any().any():
            st.error("The input features contain missing values.")
            st.stop()

        # Make predictions using the saved model
        y_pred = selected_model.predict(X_test_upload)

        # Get predicted probabilities for multiclass AUC
        y_probability = selected_model.predict_proba(
            X_test_upload
        )

        classes = list(selected_model.classes_)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test_upload, y_pred)

        precision = precision_score(
            y_test_upload,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test_upload,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test_upload,
            y_pred,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test_upload, y_pred)

        try:
            auc = roc_auc_score(
                y_test_upload,
                y_probability,
                labels=classes,
                multi_class="ovr",
                average="macro"
            )
            auc_display = f"{auc:.4f}"

        except ValueError:
            auc_display = "Not available for this test sample"

        # 7. Display evaluation metrics

        st.markdown("---")
        st.header("4. Model Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", auc_display)
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # 8. Display confusion matrix

        st.markdown("---")
        st.header("5. Confusion Matrix")

        cm = confusion_matrix(
            y_test_upload,
            y_pred,
            labels=classes
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Confusion Matrix - {selected_model_name}")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # 9. Classification report
        # ---------------------------------------

        st.markdown("---")
        st.header("6. Classification Report")

        report = classification_report(
            y_test_upload,
            y_pred,
            labels=classes,
            target_names=classes,
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ---------------------------------------
        # 10. Completion message
        # ---------------------------------------

        st.success(
            "Evaluation completed successfully! "
            "The saved model was used without retraining."
        )

    except Exception as e:

        st.error(f"An error occurred during evaluation: {e}")

else:

    st.info(
        "Please upload test_data.csv to view model evaluation results."
    )
