import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter
import shutil

st.set_page_config(layout="wide")

# Utility: Extract ZIP
def extract_zip(uploaded_zip, extract_dir="data"):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

# Utility: Bead Segmentation
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

# CUSUM change point detection
def cusum_change_points(signal, threshold_factor=5):
    mean = np.mean(signal)
    std = np.std(signal)
    threshold = threshold_factor * std
    cumsum_pos = np.zeros(len(signal))
    cumsum_neg = np.zeros(len(signal))
    change_points = []

    for i in range(1, len(signal)):
        deviation = signal[i] - mean
        cumsum_pos[i] = max(0, cumsum_pos[i-1] + deviation)
        cumsum_neg[i] = min(0, cumsum_neg[i-1] + deviation)

        if cumsum_pos[i] > threshold:
            change_points.append((i, 'Positive'))
            cumsum_pos[i] = 0
        elif cumsum_neg[i] < -threshold:
            change_points.append((i, 'Negative'))
            cumsum_neg[i] = 0

    return change_points

# Streamlit App
st.title("CUSUM-Based Change Point Inspection per Bead")

with st.sidebar:
    uploaded_zip = st.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)

    # Parameters
    seg_col = sample_df.columns[2]
    signal_col = sample_df.columns[0]
    seg_thresh = 3.0
    analysis_percent = 100
    use_smooth = True
    polyorder = 3

    with open("uploaded.zip", "wb") as f:
        f.write(uploaded_zip.getbuffer())
    files = extract_zip("uploaded.zip")

    raw_beads = defaultdict(list)
    bead_lengths = []

    for file in files:
        df = pd.read_csv(file)
        segments = segment_beads(df, seg_col, seg_thresh)
        for bead_num, (start, end) in enumerate(segments, 1):
            signal = df.iloc[start:end+1][signal_col].reset_index(drop=True)
            raw_beads[bead_num].append((os.path.basename(file), signal))
            bead_lengths.append(len(signal))

    sorted_lengths = sorted(bead_lengths)
    ratios = [sorted_lengths[i+1]/sorted_lengths[i] for i in range(len(sorted_lengths)-1)]
    max_jump_idx = np.argmax(ratios)
    split_length = sorted_lengths[max_jump_idx]
    bead_options = sorted(raw_beads.keys())

    global_summary = defaultdict(lambda: {"NOK": [], "OK_Check": []})

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            raw_length = len(raw_sig)
            win_len = min(raw_length // 20 * 2 + 1, 199)
            win_len = max(win_len, 5)

            sig = raw_sig.copy()
            if use_smooth and len(sig) >= win_len:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            change_points = cusum_change_points(sig, threshold_factor=5)
            flag = "OK"
            for idx, direction in change_points:
                if direction == "Positive":
                    flag = "NOK"
                    break
                elif direction == "Negative":
                    flag = "OK_Check"
                    break

            if flag == "NOK":
                global_summary[fname]["NOK"].append(f"{bead_num} ({bead_type})")
            elif flag == "OK_Check":
                global_summary[fname]["OK_Check"].append(f"{bead_num} ({bead_type})")

    st.subheader("Global NOK and OK_Check Beads Summary Across All Beads")
    if global_summary:
        global_table = pd.DataFrame([
            {
                "File": file,
                "NOK (Positive)": ", ".join(beads["NOK"]) if beads["NOK"] else "-",
                "NOK (Negative)": ", ".join(beads["OK_Check"]) if beads["OK_Check"] else "-"
            }
            for file, beads in global_summary.items()
        ])
        st.dataframe(global_table)
    else:
        st.write("✅ No NOK or OK_Check beads detected across all files and beads.")

    with st.sidebar:
        selected_bead = st.selectbox("Select Bead Number for Detailed Inspection", bead_options)

    raw_fig = go.Figure()
    detailed_inspection = []

    for bead_num in [selected_bead]:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            raw_length = len(raw_sig)
            win_len = min(raw_length // 20 * 2 + 1, 199)
            win_len = max(win_len, 5)

            sig = raw_sig.copy()
            if use_smooth and len(sig) >= win_len:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            change_points = cusum_change_points(sig, threshold_factor=5)

            flag = "OK"
            for idx, direction in change_points:
                if direction == "Positive":
                    flag = "NOK"
                    break
                elif direction == "Negative":
                    flag = "OK_Check"
                    break

            color = 'red' if flag == "NOK" else 'blue' if flag == "OK_Check" else 'black'

            for idx, direction in change_points:
                raw_fig.add_vline(x=idx, line_color="red", opacity=0.5)

            raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)"))
            raw_fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} (smoothed)", line=dict(color=color)))

            bead_df = pd.DataFrame({
                "File": fname,
                "Bead": bead_num,
                "Bead Type": bead_type,
                "Change Point Index": [idx for idx, _ in change_points],
                "Direction": [direction for _, direction in change_points],
                "Flag": flag
            })
            detailed_inspection.append(bead_df)

    if detailed_inspection:
        detailed_windows_df = pd.concat(detailed_inspection, ignore_index=True)
        st.subheader("Raw and Smoothed Signal with Change Points (CUSUM)")
        st.plotly_chart(raw_fig, use_container_width=True)

        st.subheader("CUSUM Detected Change Points Table")
        st.dataframe(detailed_windows_df)
    else:
        st.info("No beads to display.")
