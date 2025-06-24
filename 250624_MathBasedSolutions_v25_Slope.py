# Slope-Based Signal Anomaly Detector with session-aware state and threshold interactivity + heatmap and window slope

import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
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
def calculate_slope(signal, window_size, interval):
    if len(signal) < 2 or window_size <= 1:
        return 0, 0, 0
    slopes = []
    for start in range(0, len(signal) - window_size + 1, interval):
        x = np.arange(window_size)
        y = signal.iloc[start:start + window_size]
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return min(slopes), max(slopes), np.mean(slopes) if slopes else (0, 0, 0)

# --- Heatmap Generator ---
def generate_bead_plot_and_stats(bead_data, slope_window, slope_interval, slope_threshold):
    slope_stats = defaultdict(list)
    all_summary = []

    for bead_num, entries in bead_data.items():
        for fname, sig in entries:
            min_slope, max_slope, avg_slope = calculate_slope(sig, slope_window, slope_interval)
            is_nok = abs(max_slope) > slope_threshold
            slope_stats[bead_num].append((fname, max_slope, is_nok))
            all_summary.append({
                "File": fname,
                "Bead": bead_num,
                "Min Slope": round(min_slope, 4),
                "Max Slope": round(max_slope, 4),
                "Avg Slope": round(avg_slope, 4),
                "Window": slope_window,
                "Interval": slope_interval,
                "Threshold": slope_threshold,
                "Result": "NOK" if is_nok else "OK",
                "Reason": f"|max_slope| > {slope_threshold}" if is_nok else "OK"
            })

    selected_bead = st.selectbox("Select Bead Number to Display", sorted(slope_stats.keys()))
    fig = go.Figure()
    for fname, sig in dict(bead_data[selected_bead]).items():
        _, max_slope, _ = calculate_slope(sig, slope_window, slope_interval)
        is_nok = abs(max_slope) > slope_threshold
        color = 'red' if is_nok else 'black'
        fig.add_trace(go.Scatter(y=sig, mode='lines',
                                 name=f"{fname} | max_slope={max_slope:.4f}",
                                 line=dict(color=color)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Slope Statistics for All Beads")
    st.dataframe(pd.DataFrame(all_summary))


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Slope-Based Signal Anomaly Detector")

uploaded_file = st.file_uploader("Upload ZIP file containing CSVs", type='zip')

slope_window = st.number_input("Slope Window Size", min_value=2, value=10)
slope_interval = st.number_input("Slope Interval", min_value=1, value=5)
slope_threshold = st.number_input("Slope Threshold", value=0.5)

if uploaded_file:
    extract_dir = "extracted_data"
    csv_files = extract_zip(uploaded_file, extract_dir)

    st.success(f"{len(csv_files)} CSV files extracted.")

    filter_column = st.text_input("Bead Segmentation Column (e.g., LaserOn)", value="LaserOn")
    signal_column = st.text_input("Signal Column (e.g., Signal)", value="Signal")
    segment_threshold = st.number_input("Segmentation Threshold", value=0.5)

    if st.button("Run Analysis"):
        bead_data = process_files(csv_files, filter_column, segment_threshold, signal_column)

        if not bead_data:
            st.warning("No beads found. Check your segmentation column and threshold.")
        else:
            generate_bead_plot_and_stats(bead_data, slope_window, slope_interval, slope_threshold)
