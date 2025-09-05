import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="IntelliScore Dashboard", layout="wide")

st.title("📊 Intelliscore Dashboard")

menu = ["Overview", "Credit Scoring", "Fraud Detection"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Overview":
    st.subheader("System Overview")
    st.write("This dashboard shows results from ML models for credit scoring and fraud detection.")

elif choice == "Credit Scoring":
    st.subheader("Credit Scoring Results")
    sample_scores = np.random.rand(10) # placeholder until model is connected
    df = pd.DataFrame({"Customer ID": range(1, 11), "Credit Score": sample_scores})
    st.dataframe(df)
    st.bar_chart(df.set_index("Customer ID"))

elif choice == "Fraud Detection":
    st.subheader("Fraud Detection Alerts")
    data = {"Transaction ID": range(1, 11), "Fraud Risk": np.random.choice(["Low", "Medium", "High"], 10)}
    st.table(pd.DataFrame(data))