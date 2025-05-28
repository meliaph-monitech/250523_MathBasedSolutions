import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

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

# --- Dip Valley Detection ---
def detect_valley_mask(signal, baseline, drop_threshold, min_duration):
    in_dip = False
    dip_start = 0
    mask = np.zeros_like(signal, dtype=bool)
    for i in range(len(signal)):
        if not in_dip and signal[i] < baseline[i] - drop_threshold:
            in_dip = True
            dip_start = i
        elif in_dip and signal[i] >= baseline[i] - drop_threshold:
            dip_end = i
            if dip_end - dip_start >= min_duration:
                mask[dip_start:dip_end] = True
            in_dip = False
    if in_dip and len(signal) - dip_start >= min_duration:
        mask[dip_start:] = True
    return mask

# --- Streamlit App ---
st.set_page_config(page_title="Dip Valley Visualizer", layout="wide")
st.title("Dip Valley Detector for Laser Welding")

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
    drop_threshold = st.sidebar.slider("Drop Threshold (absolute units)", 0.01, 5.0, 0.3, 0.01)
    min_duration = st.sidebar.slider("Min duration of dip (points)", 10, 500, 50, 5)
    window_size = st.sidebar.slider("Rolling window size for baseline", 3, 101, 25, 2)

    selected_bead = st.selectbox("Select Bead Number to Display", sorted(st.session_state["bead_data"].keys()))

    if selected_bead:
        fig = go.Figure()
        signals = st.session_state["bead_data"][selected_bead]

        min_len = min(len(sig) for _, sig in signals if hasattr(sig, '__len__'))
        for file_name, signal in signals:
            signal = signal[:min_len]
            baseline = pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).median().to_numpy()
            mask = detect_valley_mask(signal, baseline, drop_threshold, min_duration)

            normal_y = np.where(mask, np.nan, signal)
            dip_y = np.where(mask, signal, np.nan)

            fig.add_trace(go.Scatter(y=normal_y, mode='lines', name=f"{file_name} (normal)", line=dict(color='black')))
            fig.add_trace(go.Scatter(y=dip_y, mode='lines', name=f"{file_name} (dip)", line=dict(color='red')))

        fig.add_trace(go.Scatter(y=baseline, mode='lines', name='Baseline', line=dict(color='green', dash='dash')))
        fig.update_layout(title=f"Dip Valley Detection - Bead #{selected_bead}", xaxis_title="Index", yaxis_title="Signal Value")
        st.plotly_chart(fig, use_container_width=True)
