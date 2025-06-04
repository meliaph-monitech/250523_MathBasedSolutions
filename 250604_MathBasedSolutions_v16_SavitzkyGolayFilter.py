import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter

# --- File Extraction ---
def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for root, _, files in os.walk(extract_dir):
            for file in files:
                os.remove(os.path.join(root, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    ok_files, test_files = [], []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                if 'ok' in root.lower():
                    ok_files.append(full_path)
                elif 'test' in root.lower():
                    test_files.append(full_path)
    return ok_files, test_files

# --- Bead Segmentation ---
def segment_beads(df, column, threshold):
    signal = df[column].to_numpy()
    start_indices, end_indices = [], []
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

# --- Apply Savitzky-Golay Filter ---
def apply_savitzky_golay(signal, window_length=7, polyorder=3, deriv=0, delta=1.0, mode='interp'):
    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return savgol_filter(signal, window_length, polyorder, deriv=deriv, delta=delta, mode=mode)

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Savitzky-Golay Filter Exploration")

st.sidebar.header("Upload ZIP File")
zip_file = st.sidebar.file_uploader("Upload a ZIP containing 'ok/' and 'test/' folders with CSVs", type="zip")

if zip_file:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)

    columns = sample_file.columns.tolist()
    filter_column = st.sidebar.selectbox("Segmentation column", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Signal column", columns)

    # Savitzky-Golay filter parameters
    window_length = st.sidebar.slider("Window Length (odd)", 3, 51, 7, step=2)
    polyorder = st.sidebar.slider("Polynomial Order", 1, 5, 3)
    deriv = st.sidebar.slider("Derivative Order", 0, 2, 0)
    delta = st.sidebar.number_input("Delta (Sample spacing)", value=1.0)
    mode = st.sidebar.selectbox("Mode", ['interp', 'mirror', 'nearest', 'constant', 'wrap'])

    if st.sidebar.button("Run Segmentation & Filtering"):
        with open("data.zip", "wb") as f:
            f.write(zip_file.getbuffer())

        ok_files, test_files = extract_zip("data.zip", "data")

        def process_files(files):
            bead_data = defaultdict(list)
            for file in files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    smoothed = apply_savitzky_golay(
                        signal,
                        window_length=window_length,
                        polyorder=polyorder,
                        deriv=deriv,
                        delta=delta,
                        mode=mode
                    )
                    bead_data[bead_num].append((os.path.basename(file), signal, smoothed))
            return bead_data

        ok_beads = process_files(ok_files)
        test_beads = process_files(test_files)

        st.session_state["ok_beads"] = ok_beads
        st.session_state["test_beads"] = test_beads
        st.success("✅ Bead segmentation and filtering completed.")

        st.markdown("### Filtered Signal Visualization")
        bead_numbers = sorted(set(ok_beads.keys()).union(test_beads.keys()))
        selected_bead = st.selectbox("Select Bead Number", bead_numbers)

        def plot_beads(beads, label, color):
            fig = go.Figure()
            for fname, raw, smooth in beads.get(selected_bead, []):
                fig.add_trace(go.Scatter(y=raw, mode='lines', name=f"Raw: {fname}", line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(y=smooth, mode='lines', name=f"{label}: {fname}", line=dict(color=color, width=2)))
            st.plotly_chart(fig, use_container_width=True)

        plot_beads(ok_beads, "Smoothed OK", "blue")
        plot_beads(test_beads, "Smoothed Test", "red")
