# Slope-Based Signal Anomaly Detector with session-aware state and threshold interactivity + heatmap and window slope

import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

# --- Process Files ---
def process_files(files, filter_column, threshold, signal_column):
    bead_data = defaultdict(list)
    for file in files:
        df = pd.read_csv(file)
        segments = segment_beads(df, filter_column, threshold)
        for bead_num, (start, end) in enumerate(segments, start=1):
            signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
            bead_data[bead_num].append((os.path.basename(file), signal))
    return bead_data

# --- Slope Calculation ---
def calculate_slope(signal, window_size):
    if len(signal) < 2:
        return 0
    if window_size > 0 and len(signal) >= window_size:
        x = np.arange(window_size)
        y = signal.iloc[:window_size]
    else:
        x = np.arange(len(signal))
        y = signal
    return np.polyfit(x, y, 1)[0]

# --- Heatmap Generator ---
def generate_heatmap(bead_data, title):
    bead_lengths = defaultdict(lambda: defaultdict(int))
    bead_nums = set()
    file_names = set()
    for bead_num, entries in bead_data.items():
        for fname, sig in entries:
            bead_lengths[fname][bead_num] = len(sig)
            bead_nums.add(bead_num)
            file_names.add(fname)
    bead_nums = sorted(bead_nums)
    file_names = sorted(file_names)
    heatmap_data = np.zeros((len(file_names), len(bead_nums)))
    for i, fname in enumerate(file_names):
        for j, bead in enumerate(bead_nums):
            heatmap_data[i, j] = bead_lengths[fname].get(bead, 0)
    df_hm = pd.DataFrame(heatmap_data, index=file_names, columns=bead_nums)
    fig, ax = plt.subplots(figsize=(max(6, len(bead_nums)), max(6, len(file_names)*0.4)))
    sns.heatmap(df_hm, annot=False, cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Bead Number")
    ax.set_ylabel("File Name")
    st.pyplot(fig)

# --- App UI Setup ---
st.set_page_config(layout="wide")
st.title("Slope-Based Signal Anomaly Detector")

uploaded_zip = st.sidebar.file_uploader("Upload ZIP file with test signals", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Signal column for slope analysis", columns)

    if st.sidebar.button("Segment Beads"):
        with open("data.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extracted_files = extract_zip("data.zip", "slope_data")
        bead_data = process_files(extracted_files, filter_column, threshold, signal_column)
        st.session_state["bead_data"] = bead_data
        st.session_state["analysis_ready"] = True

    st.markdown("### Bead Length Heatmap")
    generate_heatmap(st.session_state["bead_data"], "Bead Lengths per File")

if "bead_data" in st.session_state and st.session_state.get("analysis_ready", False):
    bead_data = st.session_state["bead_data"]
    slope_threshold = st.sidebar.slider("Slope Threshold (absolute)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    slope_window = st.sidebar.number_input("Slope Calculation Window (points)", min_value=0, max_value=5000, value=100, step=10)

    all_summary = []
    slope_stats = defaultdict(list)

    for bead_num, entries in bead_data.items():
        for fname, sig in entries:
            slope = calculate_slope(sig, slope_window)
            is_nok = abs(slope) > slope_threshold
            slope_stats[bead_num].append((fname, slope, is_nok))
            all_summary.append({
                "File": fname,
                "Bead": bead_num,
                "Slope": round(slope, 4),
                "Window": slope_window,
                "Threshold": slope_threshold,
                "Result": "NOK" if is_nok else "OK",
                "Reason": f"|slope| > {slope_threshold}"
            })

    selected_bead = st.selectbox("Select Bead Number to Display", sorted(slope_stats.keys()))

    fig = go.Figure()
    for fname, sig in dict(bead_data[selected_bead]).items():
        slope = calculate_slope(sig, slope_window)
        is_nok = abs(slope) > slope_threshold
        color = 'red' if is_nok else 'black'
        fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} | slope={slope:.4f}", line=dict(color=color)))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bead Length Heatmap")
    generate_heatmap(bead_data, "Bead Lengths per File")

    st.markdown("### Slope Statistics for All Beads")
    st.dataframe(pd.DataFrame(all_summary))
