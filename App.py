import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Page Setup
st.set_page_config(page_title=" Network Anomaly Detection", layout="wide")

# Load Wireshark CSV Data
df = pd.read_csv("bigdata2.csv")

# Normalize time to start from zero
df['Time'] = df['Time'] - df['Time'].min()

# Select only numerical features
df_numeric = df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

time_values = df['Time'].values  # Extract time for plotting

# Split Data for Training
train_size = st.sidebar.slider("Training Data Percentage", 50, 90, 80)
split_idx = int((train_size / 100) * len(df_scaled))
df_train = df_scaled[:split_idx]
df_test = df_scaled[split_idx:]

# Sidebar for Model Parameters
st.sidebar.header("🔧 Model Parameters")
contamination_rate = st.sidebar.slider("Anomaly Contamination Rate (%)", 1, 10, 5) / 100
n_clusters = st.sidebar.slider("K-Means Clusters", 2, 5, 2)

def detect_anomalies():
    # Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    iso_forest.fit(df_train)
    iso_scores = -iso_forest.decision_function(df_test)
    iso_threshold = np.percentile(iso_scores, 95)
    iso_anomalies = (iso_scores > iso_threshold).astype(int)
    
    # One-Class SVM
    oc_svm = OneClassSVM(nu=contamination_rate, kernel="rbf")
    oc_svm.fit(df_train)
    oc_scores = -oc_svm.decision_function(df_test)
    oc_threshold = np.percentile(oc_scores, 95)
    oc_anomalies = (oc_scores > oc_threshold).astype(int)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_train)
    distances = np.min(kmeans.transform(df_test), axis=1)
    kmeans_threshold = np.percentile(distances, 95)
    kmeans_anomalies = (distances > kmeans_threshold).astype(int)
    
    # Autoencoder
    input_dim = df_train.shape[1]
    autoencoder = keras.Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(df_train, df_train, epochs=30, batch_size=32, verbose=0)
    
    reconstructions = autoencoder.predict(df_test)
    reconstruction_error = np.mean(np.abs(df_test - reconstructions), axis=1)
    autoencoder_threshold = np.percentile(reconstruction_error, 95)
    autoencoder_anomalies = (reconstruction_error > autoencoder_threshold).astype(int)
    
    # Majority Voting
    final_anomalies = (iso_anomalies + oc_anomalies + kmeans_anomalies + autoencoder_anomalies) >= 2
    
    return time_values[split_idx:], reconstruction_error, final_anomalies, iso_anomalies, oc_anomalies, kmeans_anomalies, autoencoder_anomalies

# Run Anomaly Detection
time_test, anomaly_scores, anomalies, iso_flags, oc_flags, kmeans_flags, auto_flags = detect_anomalies()

# Ensure the graph starts from (0,0)
time_test_with_zero = np.insert(time_test, 0, 0)
anomaly_scores_with_zero = np.insert(anomaly_scores, 0, 0)

# Dashboard Title
st.title("🚀 AI-Powered Network Anomaly Detection Dashboard")
st.markdown("Monitor **network traffic anomalies** in real-time using **AI & ML models**.")

# Graph: Anomaly Scores Over Time
st.subheader("📈 Anomaly Scores Over Time")
fig = px.line(
    x=time_test_with_zero,
    y=anomaly_scores_with_zero,
    labels={"x": "Time (seconds)", "y": "Anomaly Score"},
    title="Anomaly Detection Results",
    template="plotly_dark"
)

fig.add_trace(go.Scatter(
    x=time_test[anomalies == 1],
    y=anomaly_scores[anomalies == 1],
    mode='markers',
    marker=dict(color='red', size=8, symbol='circle'),
    name='Detected Anomalies'
))
st.plotly_chart(fig, use_container_width=True)

# Sidebar Model Selection
st.sidebar.write("### 📌 Select Anomaly Detection Model:")
technique = st.sidebar.radio("", ["Isolation Forest", "One-Class SVM", "K-Means", "Autoencoder"])

# Model Flag Mapping
technique_map = {
    "Isolation Forest": iso_flags,
    "One-Class SVM": oc_flags,
    "K-Means": kmeans_flags,
    "Autoencoder": auto_flags
}

# Display Model-Specific Results
st.subheader(f"📍 {technique} Results")
st.write(f"🔍 **Anomalies Detected:** {int(sum(anomalies))}")

# Anomaly Distribution Heatmap
st.subheader("🔥 Anomaly Heatmap")
anomaly_df = pd.DataFrame({"Time": time_test, "Anomaly": anomalies})
heatmap_fig = px.scatter(
    anomaly_df, x="Time", y=anomalies,
    color=anomalies, color_continuous_scale="reds",
    title="Anomaly Density Over Time"
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# Detailed Anomaly Info
if st.checkbox("📜 Show Detailed Anomalies"):
    detailed_anomalies = pd.DataFrame({
        "Time": time_test[anomalies == 1],
        "Anomaly Score": anomaly_scores[anomalies == 1]
    })
    st.dataframe(detailed_anomalies.style.highlight_max(axis=0))

# Summary Statistics
st.sidebar.subheader("📊 Anomaly Summary")
st.sidebar.metric("Total Anomalies", int(sum(anomalies)))
st.sidebar.metric("Peak Anomaly Score", round(np.max(anomaly_scores), 4))

# Histogram of Anomaly Scores
st.subheader("📊 Histogram of Anomaly Scores")
hist_fig = px.histogram(
    anomaly_scores, nbins=50, labels={"x": "Anomaly Score", "y": "Frequency"},
    title="Distribution of Anomaly Scores", template="plotly_dark"
)
st.plotly_chart(hist_fig, use_container_width=True)
