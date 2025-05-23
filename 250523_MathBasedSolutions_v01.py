import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from collections import defaultdict

# --- File Extraction ---
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]


# --- Bead Segmentation ---
def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))


# --- Feature Extraction ---
def extract_features(signal):
    if len(signal) == 0:
        return {}

    n = len(signal)
    fft_vals = fft(signal)
    psd = np.abs(fft_vals)**2
    psd_norm = psd[:n//2] / np.sum(psd[:n//2]) if np.sum(psd[:n//2]) > 0 else np.zeros(n//2)

    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'peak_to_peak': np.ptp(signal),
        'energy': np.sum(signal**2),
        'rms': np.sqrt(np.mean(signal**2)),
        'slope': np.polyfit(np.arange(n), signal, 1)[0],
        'entropy': -np.sum(psd_norm * np.log2(psd_norm + 1e-12)),
        'autocorrelation': np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0,
    }


# --- Rule-Based Evaluation ---
def evaluate_rules(features, rules, logic_mode):
    violations = []
    for feature, bounds in rules.items():
        val = features.get(feature, None)
        if val is None:
            continue
        min_val, max_val = bounds
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            violations.append(True)
        else:
            violations.append(False)

    return any(violations) if logic_mode == 'any' else all(violations)


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection (Math Rule-Based)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP with CSVs", type="zip")
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("LO threshold", value=0.0)
        signal_column = st.selectbox("Signal column for feature extraction", [col for col in columns if df_sample[col].dtype in [np.float64, np.int64]])

        rule_logic = st.radio("Rule logic", ["any", "all"], format_func=lambda x: "Any rule violated = NOK" if x == "any" else "All rules must be violated = NOK")

        # Threshold inputs for each feature
        feature_rules = {}
        for feature in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis', 'peak_to_peak', 'energy', 'rms', 'slope', 'entropy', 'autocorrelation']:
            with st.expander(f"Set thresholds for {feature}"):
                min_val = st.number_input(f"{feature} - Min", value=None, format="%.5f", key=f"{feature}_min")
                max_val = st.number_input(f"{feature} - Max", value=None, format="%.5f", key=f"{feature}_max")
                feature_rules[feature] = (min_val if min_val != 0 else None, max_val if max_val != 0 else None)

        if st.button("Run Analysis"):
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df[signal_column].iloc[start:end+1].values
                    features = extract_features(signal)
                    status = "NOK" if evaluate_rules(features, feature_rules, rule_logic) else "OK"
                    metadata.append({"file": file, "bead_number": bead_num, "status": status, "start": start, "end": end})

            st.session_state["results"] = metadata

# --- Results Display ---
if "results" in st.session_state:
    results_df = pd.DataFrame(st.session_state["results"])
    st.subheader("Detection Results Table")
    st.dataframe(results_df[["file", "bead_number", "status"]])

    st.subheader("Visualization")
    bead_numbers = sorted(results_df["bead_number"].unique())
    selected_bead = st.selectbox("Select Bead Number", bead_numbers)

    fig = go.Figure()
    for _, row in results_df[results_df["bead_number"] == selected_bead].iterrows():
        df = pd.read_csv(row["file"])
        signal = df[signal_column].iloc[row["start"]:row["end"]+1].values
        color = "red" if row["status"] == "NOK" else "black"
        fig.add_trace(go.Scatter(y=signal, mode="lines", line=dict(color=color, width=1), name=f"{row['file']} ({row['status']})"))

    fig.update_layout(title=f"Bead #{selected_bead} Signal Comparison", xaxis_title="Index", yaxis_title=signal_column)
    st.plotly_chart(fig, use_container_width=True)
