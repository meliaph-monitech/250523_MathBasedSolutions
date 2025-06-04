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
zip1 = st.sidebar.file_uploader("ZIP 1: Defocusing", type="zip")
zip2 = st.sidebar.file_uploader("ZIP 2: GAP", type="zip")
zip3 = st.sidebar.file_uploader("ZIP 3: OK", type="zip")

if zip1 and zip2 and zip3:
    with zipfile.ZipFile(zip1, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Select column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Select signal column for analysis", columns)

    if st.sidebar.button("Segment Beads"):
        with open("zip1.zip", "wb") as f:
            f.write(zip1.getbuffer())
        with open("zip2.zip", "wb") as f:
            f.write(zip2.getbuffer())
        with open("zip3.zip", "wb") as f:
            f.write(zip3.getbuffer())

        zip1_files = extract_zip("zip1.zip", "data_zip1")
        zip2_files = extract_zip("zip2.zip", "data_zip2")
        zip3_files = extract_zip("zip3.zip", "data_zip3")

        def process_files(files):
            bead_data = defaultdict(list)
            for file in files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    bead_data[bead_num].append((os.path.basename(file), signal))
            return bead_data

        st.session_state["zip1_beads_raw"] = process_files(zip1_files)
        st.session_state["zip2_beads_raw"] = process_files(zip2_files)
        st.session_state["zip3_beads_raw"] = process_files(zip3_files)
        st.success("✅ Bead segmentation completed.")

# --- Filtering and Display ---
if all(k in st.session_state for k in ["zip1_beads_raw", "zip2_beads_raw", "zip3_beads_raw"]):
    st.sidebar.header("Savitzky-Golay Filtering")
    window_length = st.sidebar.slider("Window Length", 3, 51, 7, step=2)
    polyorder = st.sidebar.slider("Polynomial Order", 1, 5, 3)

    # Display options
    st.sidebar.header("Display Options")
    show_zip1 = st.sidebar.checkbox("Show ZIP Defocusing Data", value=True)
    show_zip2 = st.sidebar.checkbox("Show ZIP GAP Data", value=True)
    show_zip3 = st.sidebar.checkbox("Show ZIP OK Data", value=True)
    show_raw = st.sidebar.checkbox("Show Raw Signal", value=True)
    show_filtered = st.sidebar.checkbox("Show Filtered Signal", value=True)

    # Apply filter to raw beads
    def filter_beads(beads_raw):
        beads_filtered = defaultdict(list)
        for bead_num, records in beads_raw.items():
            for fname, signal in records:
                smoothed = apply_savitzky_golay(signal, window_length, polyorder)
                beads_filtered[bead_num].append((fname, signal, smoothed))
        return beads_filtered

    zip1_beads = filter_beads(st.session_state["zip1_beads_raw"])
    zip2_beads = filter_beads(st.session_state["zip2_beads_raw"])
    zip3_beads = filter_beads(st.session_state["zip3_beads_raw"])

    all_beads = sorted(set(zip1_beads.keys()) | set(zip2_beads.keys()) | set(zip3_beads.keys()))
    st.markdown("### Filtered Signal Visualization")
    selected_bead = st.selectbox("Select Bead Number to Display", all_beads)

    fig = go.Figure()

    if show_zip1 and selected_bead in zip1_beads:
        for fname, raw_signal, smoothed_signal in zip1_beads[selected_bead]:
            if show_raw:
                fig.add_trace(go.Scatter(y=raw_signal, mode='lines', name=f"Raw ZIP1: {fname}", line=dict(color='gray', width=1)))
            if show_filtered:
                fig.add_trace(go.Scatter(y=smoothed_signal, mode='lines', name=f"Filtered ZIP1: {fname}", line=dict(color='blue', width=2)))

    if show_zip2 and selected_bead in zip2_beads:
        for fname, raw_signal, smoothed_signal in zip2_beads[selected_bead]:
            if show_raw:
                fig.add_trace(go.Scatter(y=raw_signal, mode='lines', name=f"Raw ZIP2: {fname}", line=dict(color='lightgray', width=1)))
            if show_filtered:
                fig.add_trace(go.Scatter(y=smoothed_signal, mode='lines', name=f"Filtered ZIP2: {fname}", line=dict(color='red', width=2)))

    if show_zip3 and selected_bead in zip3_beads:
        for fname, raw_signal, smoothed_signal in zip3_beads[selected_bead]:
            if show_raw:
                fig.add_trace(go.Scatter(y=raw_signal, mode='lines', name=f"Raw ZIP3: {fname}", line=dict(color='darkgray', width=1)))
            if show_filtered:
                fig.add_trace(go.Scatter(y=smoothed_signal, mode='lines', name=f"Filtered ZIP3: {fname}", line=dict(color='green', width=2)))

    if not fig.data:
        st.warning("No data to display. Please check your display options.")
    else:
        st.plotly_chart(fig, use_container_width=True)
