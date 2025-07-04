import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = joblib.load("model/threat_detection_model.pkl")

st.title("🛡️ AI Threat Detection System")
st.markdown("Upload a CSV file (with same features as training set) to detect threats.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(data.head())

    # Predict button
    if st.button("Predict"):
        try:
            # Make predictions
            predictions = model.predict(data)

            # Display results
            st.subheader("✅ Prediction Results")
            data['Prediction'] = predictions
            st.dataframe(data[['Prediction']])

            # Show accuracy (if actual labels exist)
            if 'label' in data.columns:
                accuracy = accuracy_score(data['label'], predictions)
                st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

                # Confusion Matrix
                cm = confusion_matrix(data['label'], predictions)
                st.subheader("📊 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # Classification report
                st.subheader("📋 Classification Report")
                st.text(classification_report(data['label'], predictions))

            # Download predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {str(e)}")
