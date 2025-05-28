import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from scipy.stats import zscore
from scipy.signal import correlate

# --- File Extraction ---
def extract_zip(uploaded_file, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()

    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the ZIP file.")
        st.stop()

    return [os.path.join(extract_dir, f) for f in csv_files]

# --- Bead Segmentation ---
def segment_beads(df, column, threshold):
    start_indices = []
    end_indices = []
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

# --- Scoring Anomalies ---
def compute_baseline(signals):
    return np.median(signals, axis=0)

def compute_rms(signal, window_size):
    return np.sqrt(pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()**2).to_numpy()

def compute_correlation_to_baseline(signal, baseline):
    corr = np.corrcoef(signal, baseline)[0, 1]
    return corr

# --- Streamlit App ---
st.set_page_config(page_title="Advanced Dip Analyzer", layout="wide")
st.title("Behavioral Anomaly Viewer for Welding Signals V11")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])

    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())

        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        signal_column = st.selectbox("Select signal column for analysis", columns)

        if st.button("Segment Beads"):
            bead_data = defaultdict(list)
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    bead_data[bead_num].append((os.path.basename(file), signal))

            st.session_state["bead_data"] = bead_data
            st.session_state["signal_column"] = signal_column
            st.success("Bead segmentation completed.")

if "bead_data" in st.session_state:
    selected_bead = st.selectbox("Select Bead Number to Display", sorted(st.session_state["bead_data"].keys()))

    signals = st.session_state["bead_data"][selected_bead]
    min_len = min(len(sig) for _, sig in signals if hasattr(sig, '__len__'))
    trimmed_signals = [sig[:min_len] for _, sig in signals]
    stacked = np.vstack(trimmed_signals)
    baseline = compute_baseline(stacked)

    # Precompute dynamic thresholds
    correlations = [compute_correlation_to_baseline(sig[:min_len], baseline) for _, sig in signals]
    all_energy_zscores = []
    default_rms_win = max(10, min_len // 10)
    for _, sig in signals:
        energy = compute_rms(sig[:min_len], default_rms_win)
        baseline_energy = compute_rms(baseline, default_rms_win)
        all_energy_zscores.extend(zscore(energy - baseline_energy))

    min_corr, max_corr = round(min(correlations), 2), round(max(correlations), 2)
    min_z, max_z = round(min(all_energy_zscores), 2), round(max(all_energy_zscores), 2)

    rms_window = st.sidebar.slider("RMS Energy Window Size", 5, min_len, default_rms_win, 5)
    correlation_threshold = st.sidebar.slider("Min Correlation to Baseline (to be OK)", min_corr, 1.0, round(np.median(correlations), 2), 0.01)
    energy_threshold = st.sidebar.slider("Z-score Energy Drop (to be NOK)", min_z, 0.0, round(np.percentile(all_energy_zscores, 10), 2), 0.1)

    fig = go.Figure()
    summary = []

    for (file_name, signal) in signals:
        signal = signal[:min_len]
        corr = compute_correlation_to_baseline(signal, baseline)
        signal_energy = compute_rms(signal, rms_window)
        baseline_energy = compute_rms(baseline, rms_window)
        energy_diff_z = zscore(signal_energy - baseline_energy)

        dip_mask = energy_diff_z < energy_threshold
        normal_y = np.where(dip_mask, np.nan, signal)
        dip_y = np.where(dip_mask, signal, np.nan)
        color = 'red' if corr < correlation_threshold else 'black'

        fig.add_trace(go.Scatter(y=normal_y, mode='lines', name=f"{file_name} (normal)", line=dict(color=color)))
        fig.add_trace(go.Scatter(y=dip_y, mode='lines', name=f"{file_name} (dip)", line=dict(color='orange')))

        summary.append({
            "File": file_name,
            "Correlation": round(corr, 3),
            "Min Energy Z": round(np.min(energy_diff_z), 2),
            "Potential NOK": corr < correlation_threshold and np.min(energy_diff_z) < energy_threshold
        })

    fig.add_trace(go.Scatter(y=baseline, mode='lines', name='Baseline (median)', line=dict(color='green', dash='dash')))
    fig.update_layout(title=f"Bead #{selected_bead} - Behavioral Anomaly Detection", xaxis_title="Index", yaxis_title="Signal Value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Table")
    st.dataframe(pd.DataFrame(summary))
