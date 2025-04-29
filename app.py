

import streamlit as st
import pickle
import pandas as pd
# Load the saved scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('hierarchical.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# App title and description
st.set_page_config(page_title="Customer Segmentation App", page_icon="💼", layout="centered")

st.title("💼 Customer Segmentation Dashboard")
st.subheader("🚀 Predict Customer Segment Using KMeans Clustering")

st.markdown("""
Welcome to the Customer Segmentation App!  
You can either manually enter customer details or upload a CSV file to predict customer clusters.  
This helps businesses target customers more smartly.
""")

# Sidebar options
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["Manual Entry", "Upload CSV"])

# Define the feature order
features = ['Age', 'CustAccountBalance', 'TotalTransactionAmount', 'AvgTransactionAmount', 'TransactionCount']

# Manual Entry
if app_mode == "Manual Entry":
    st.header("📋 Enter Customer Details Manually:")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    balance = st.number_input("Customer Account Balance", value=10000.0)
    total_amount = st.number_input("Total Transaction Amount", value=5000.0)
    avg_amount = st.number_input("Average Transaction Amount", value=250.0)
    txn_count = st.number_input("Transaction Count", value=10)

    input_data = pd.DataFrame([[age, balance, total_amount, avg_amount, txn_count]], columns=features)

    if st.button("Predict Cluster"):
        scaled_input = scaler.transform(input_data)
        cluster_label = kmeans_model.predict(scaled_input)[0]
        
        st.success(f"Predicted Customer Segment: **Cluster {cluster_label}** 🧩")

        # Add extra interpretation if needed
        if cluster_label == 0:
            st.info("💡 Cluster 0: Young high spenders")
        elif cluster_label == 1:
            st.info("💡 Cluster 1: Mature customers with stable transactions")
        elif cluster_label == 2:
            st.info("💡 Cluster 2: Low activity / low balance customers")
        else:
            st.info("💡 Cluster 3: Premium high net worth individuals")

# Upload CSV Mode
elif app_mode == "Upload CSV":
    st.header("📤 Upload CSV File:")

    uploaded_file = st.file_uploader("Choose a CSV file with correct columns", type=['csv'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Check columns
            if all(col in data.columns for col in features):
                st.success("CSV looks good! Now predicting clusters...")
                
                scaled_data = scaler.transform(data[features])
                cluster_preds = kmeans_model.predict(scaled_data)
                
                data['Predicted_Cluster'] = cluster_preds
                st.dataframe(data)

                st.download_button("Download Predictions", data.to_csv(index=False), "clustered_customers.csv", "text/csv")

            else:
                st.error(f"CSV must contain the columns: {features}")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by a Future Data Scientist")
