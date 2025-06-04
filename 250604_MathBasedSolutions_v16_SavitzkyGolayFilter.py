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
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

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

# --- Apply Savitzky-Golay Filter ---
def apply_savitzky_golay(signal, window_length=7, polyorder=3):
    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return savgol_filter(signal, window_length, polyorder)

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Savitzky-Golay Filter Exploration")

st.sidebar.header("Upload Data")
ok_zip = st.sidebar.file_uploader("ZIP file with ONLY OK welds", type="zip")
test_zip = st.sidebar.file_uploader("ZIP file to Test", type="zip")

if ok_zip and test_zip:
    with zipfile.ZipFile(ok_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Select column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Select signal column for analysis", columns)

    window_length = st.sidebar.slider("Savitzky-Golay Window Length", 3, 51, 7, step=2)
    polyorder = st.sidebar.slider("Savitzky-Golay Polynomial Order", 1, 5, 3)

    # Display filters
    st.sidebar.header("Display Options")
    show_ok = st.sidebar.checkbox("Show OK Welds (ZIP 1)", value=True)
    show_test = st.sidebar.checkbox("Show Test Welds (ZIP 2)", value=True)
    show_raw = st.sidebar.checkbox("Show Raw Signal", value=True)
    show_filtered = st.sidebar.checkbox("Show Filtered Signal", value=True)

    if st.sidebar.button("Segment Beads"):
        with open("ok.zip", "wb") as f:
            f.write(ok_zip.getbuffer())
        with open("test.zip", "wb") as f:
            f.write(test_zip.getbuffer())

        ok_files = extract_zip("ok.zip", "ok_data")
        test_files = extract_zip("test.zip", "test_data")

        def process_files(files):
            bead_data = defaultdict(list)
            for file in files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    smoothed_signal = apply_savitzky_golay(signal, window_length, polyorder)
                    bead_data[bead_num].append((os.path.basename(file), signal, smoothed_signal))
            return bead_data

        ok_beads = process_files(ok_files)
        test_beads = process_files(test_files)

        st.session_state["ok_beads"] = ok_beads
        st.session_state["test_beads"] = test_beads

        st.success("✅ Bead segmentation completed.")

# --- Visualization ---
if "ok_beads" in st.session_state and "test_beads" in st.session_state:
    ok_beads = st.session_state["ok_beads"]
    test_beads = st.session_state["test_beads"]
    all_beads = sorted(set(ok_beads.keys()).union(test_beads.keys()))

    st.markdown("### Filtered Signal Visualization")
    selected_bead = st.selectbox("Select Bead Number to Display", all_beads)

    fig = go.Figure()

    if show_ok and selected_bead in ok_beads:
        for fname, raw_signal, smoothed_signal in ok_beads[selected_bead]:
            if show_raw:
                fig.add_trace(go.Scatter(
                    y=raw_signal,
                    mode='lines',
                    name=f"Raw OK: {fname}",
                    line=dict(color='gray', width=1)
                ))
            if show_filtered:
                fig.add_trace(go.Scatter(
                    y=smoothed_signal,
                    mode='lines',
                    name=f"Smoothed OK: {fname}",
                    line=dict(color='blue', width=2)
                ))

    if show_test and selected_bead in test_beads:
        for fname, raw_signal, smoothed_signal in test_beads[selected_bead]:
            if show_raw:
                fig.add_trace(go.Scatter(
                    y=raw_signal,
                    mode='lines',
                    name=f"Raw Test: {fname}",
                    line=dict(color='lightgray', width=1)
                ))
            if show_filtered:
                fig.add_trace(go.Scatter(
                    y=smoothed_signal,
                    mode='lines',
                    name=f"Smoothed Test: {fname}",
                    line=dict(color='red', width=2)
                ))

    if not fig.data:
        st.warning("No data to display. Please check your display options.")
    else:
        st.plotly_chart(fig, use_container_width=True)
