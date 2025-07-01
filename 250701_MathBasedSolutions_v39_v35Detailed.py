import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from scipy.signal import savgol_filter
import shutil

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

# Utility: Change Point Detection
def analyze_change_points(signal, window_size, step_size, metric, threshold, mode):
    signal = signal.dropna().reset_index(drop=True)
    change_points = []
    abs_scores, rel_scores = [], []
    positions = []
    for start in range(0, len(signal) - 2 * window_size + 1, step_size):
        curr = signal[start:start + window_size]
        next_ = signal[start + window_size:start + 2 * window_size]
        if metric == "Mean":
            v1, v2 = curr.mean(), next_.mean()
        elif metric == "Median":
            v1, v2 = curr.median(), next_.median()
        elif metric == "Standard Deviation":
            v1, v2 = curr.std(), next_.std()
        abs_diff = abs(v1 - v2)
        rel_diff = abs_diff / max(abs(v1), 1e-6)
        abs_scores.append(abs_diff)
        rel_scores.append(rel_diff)
        positions.append(start + window_size)
        check_diff = abs_diff if mode == "Absolute" else rel_diff
        if check_diff > threshold:
            change_points.append((start, start + 2 * window_size - 1, check_diff))
    return {
        "positions": positions,
        "abs_scores": abs_scores,
        "rel_scores": rel_scores,
        "change_points": change_points
    }

# Streamlit App
st.title("Detailed Change Point Inspection per Bead")

uploaded_zip = st.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    seg_col = sample_df.columns[2]
    signal_col = sample_df.columns[0]

    seg_thresh = 3.0
    analysis_percent = 100
    alu_ignore_thresh = 3.0
    cu_ignore_thresh = 3.0
    use_smooth = True
    win_len = 199
    polyorder = 5
    win_size = 350
    step_size = 175
    metric = "Median"
    mode = "Relative (%)"
    thresh_input = "15"
    threshold = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)

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
    selected_bead = st.selectbox("Select Bead Number for Detailed Inspection", bead_options)

    detailed_inspection = []
    for bead_num in [selected_bead]:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            sig = raw_sig.copy()
            if bead_type == "Aluminum":
                sig = np.minimum(sig, alu_ignore_thresh)
            else:
                sig = np.minimum(sig, cu_ignore_thresh)
            if use_smooth and len(sig) >= win_size:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))
            signal_clean = sig.dropna().reset_index(drop=True)
            records = []
            for start in range(0, len(signal_clean) - 2 * win_size + 1, step_size):
                curr = signal_clean[start:start + win_size]
                next_ = signal_clean[start + win_size:start + 2 * win_size]
                if metric == "Mean":
                    v1, v2 = curr.mean(), next_.mean()
                elif metric == "Median":
                    v1, v2 = curr.median(), next_.median()
                elif metric == "Standard Deviation":
                    v1, v2 = curr.std(), next_.std()
                abs_diff = abs(v1 - v2)
                rel_diff = abs_diff / max(abs(v1), 1e-6)
                check_diff = abs_diff if mode == "Absolute" else rel_diff
                triggered = check_diff > threshold
                records.append({
                    "File": fname,
                    "Bead": bead_num,
                    "Bead Type": bead_type,
                    "Start Index": start,
                    "End Index": start + 2 * win_size - 1,
                    "Metric Window 1": v1,
                    "Metric Window 2": v2,
                    "Abs Diff": abs_diff,
                    "Rel Diff (%)": rel_diff * 100,
                    "Threshold": threshold * 100 if mode == "Relative (%)" else threshold,
                    "Triggered Change Point": triggered
                })
            bead_df = pd.DataFrame(records)
            detailed_inspection.append(bead_df)

    detailed_windows_df = pd.concat(detailed_inspection, ignore_index=True)
    st.subheader("Detailed Per-Window Change Point Analysis")
    st.dataframe(detailed_windows_df)

    df_vis = detailed_windows_df.copy()
    df_vis['Color'] = np.where(df_vis['Triggered Change Point'], 'Triggered', 'Not Triggered')

    fig = px.scatter(
        df_vis,
        x="Start Index",
        y="Rel Diff (%)" if mode == "Relative (%)" else "Abs Diff",
        color="Color",
        hover_data=["File", "Bead", "Bead Type", "Metric Window 1", "Metric Window 2", "Threshold"],
        title=f"Window-based Change Detection for Bead {selected_bead}"
    )
    fig.add_hline(
        y=threshold*100 if mode=="Relative (%)" else threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text="Threshold",
        annotation_position="top left"
    )
    st.plotly_chart(fig, use_container_width=True)
