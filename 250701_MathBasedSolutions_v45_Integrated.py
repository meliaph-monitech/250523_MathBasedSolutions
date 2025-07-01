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

# Utility: Change Point Detection
def analyze_change_points(signal, window_size, step_size, metric, threshold):
    signal = signal.reset_index(drop=True) if isinstance(signal, pd.Series) else pd.Series(signal)
    change_points = []
    diff_scores, rel_scores, positions = [], [], []
    for start in range(0, len(signal) - 2 * window_size + 1, step_size):
        curr = signal[start:start + window_size]
        next_ = signal[start + window_size:start + 2 * window_size]
        if metric == "Mean":
            v1, v2 = curr.mean(), next_.mean()
        elif metric == "Median":
            v1, v2 = curr.median(), next_.median()
        elif metric == "Standard Deviation":
            v1, v2 = curr.std(), next_.std()
        diff = v2 - v1
        rel_diff = diff / max(abs(v1), 1e-6)
        diff_scores.append(diff)
        rel_scores.append(rel_diff)
        positions.append(start + window_size)
        if abs(rel_diff) > threshold:
            change_points.append((start, start + 2 * window_size - 1, rel_diff))
    return {
        "positions": positions,
        "diff_scores": diff_scores,
        "rel_scores": rel_scores,
        "change_points": change_points
    }

# Streamlit App
st.title("Detailed Change Point Inspection per Bead with Transparent Recheck")

with st.sidebar:
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
    use_smooth = True
    polyorder = 5
    metric = "Median"
    thresh_input = "15"
    threshold = float(thresh_input) / 100

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

    summary_table = []

    bead_flags_per_file = defaultdict(lambda: defaultdict(dict))

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            raw_length = len(raw_sig)
            win_len = min(raw_length // 20 * 2 + 1, 199)
            win_len = max(win_len, 5)
            win_size = max(raw_length // 10, 50)
            step_size = max(win_size // 2, 10)

            sig = raw_sig.copy()
            if use_smooth and len(sig) >= win_len:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            result = analyze_change_points(sig, win_size, step_size, metric, threshold)
            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)
            cp_in_region_initial = [cp for cp in result["change_points"] if cp[1] < nok_region_limit]

            flag_initial = "OK"
            for cp in cp_in_region_initial:
                if cp[2] > threshold:
                    flag_initial = "NOK"
                    break
                elif cp[2] < -threshold:
                    flag_initial = "OK_Check"
                    break

            # Transparent recheck with masking
            flag_corrected = flag_initial
            clip_threshold = np.percentile(raw_sig, 75)
            masked_sig = np.where(raw_sig > clip_threshold, clip_threshold, raw_sig)
            if use_smooth and len(masked_sig) >= win_len:
                masked_sig = savgol_filter(masked_sig, win_len, polyorder)
            masked_sig = pd.Series(masked_sig)

            result_recheck = analyze_change_points(masked_sig, win_size, step_size, metric, threshold)
            cp_in_region_recheck = [cp for cp in result_recheck["change_points"] if cp[1] < nok_region_limit]

            has_nok_recheck = any(cp[2] > threshold for cp in cp_in_region_recheck)
            has_ok_check_recheck = any(cp[2] < -threshold for cp in cp_in_region_recheck)

            if flag_initial == "NOK" and not has_nok_recheck:
                flag_corrected = "OK"
            elif flag_initial == "OK_Check" and not has_ok_check_recheck:
                flag_corrected = "OK"

            summary_table.append({
                "File": fname,
                "Bead": bead_num,
                "Bead Type": bead_type,
                "Initial NOK": "✅" if flag_initial == "NOK" else "",
                "Initial OK_Check": "✅" if flag_initial == "OK_Check" else "",
                "Corrected NOK": "✅" if flag_corrected == "NOK" else "",
                "Corrected OK_Check": "✅" if flag_corrected == "OK_Check" else ""
            })

            bead_flags_per_file[fname][bead_num] = {
                "flag_initial": flag_initial,
                "flag_corrected": flag_corrected,
                "masked_sig": masked_sig
            }

    st.subheader("Full Transparent Summary Table")
    summary_df = pd.DataFrame(summary_table)
    st.dataframe(summary_df)

    with st.sidebar:
        selected_bead = st.selectbox("Select Bead Number for Detailed Inspection", bead_options)

    raw_fig = go.Figure()
    score_fig = go.Figure()
    detailed_inspection = []

    for bead_num in [selected_bead]:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            flags = bead_flags_per_file[fname][bead_num]
            flag_initial = flags["flag_initial"]
            flag_corrected = flags["flag_corrected"]
            plot_sig = flags["masked_sig"] if flag_corrected != "OK" else raw_sig

            color = 'red' if flag_corrected == "NOK" else 'blue' if flag_corrected == "OK_Check" else 'black'

            raw_length = len(raw_sig)
            win_len = min(raw_length // 20 * 2 + 1, 199)
            win_len = max(win_len, 5)
            win_size = max(raw_length // 10, 50)
            step_size = max(win_size // 2, 10)

            sig = raw_sig.copy()
            if use_smooth and len(sig) >= win_len:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            result = analyze_change_points(sig, win_size, step_size, metric, threshold)

            raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)"))
            raw_fig.add_trace(go.Scatter(y=plot_sig, mode='lines', name=f"{fname} (masked/smoothed)", line=dict(color=color)))

            for start, end, _ in result["change_points"]:
                raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, layer="below")

            y_scores = [v * 100 for v in result["rel_scores"]]
            score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Rel Diff (%)"))

            signal_clean = pd.Series(plot_sig).reset_index(drop=True)
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
                diff = v2 - v1
                rel_diff = diff / max(abs(v1), 1e-6)
                triggered = abs(rel_diff) > threshold
                records.append({
                    "File": fname,
                    "Bead": bead_num,
                    "Bead Type": bead_type,
                    "Start Index": start,
                    "End Index": start + 2 * win_size - 1,
                    "Metric Window 1": v1,
                    "Metric Window 2": v2,
                    "Diff": diff,
                    "Rel Diff (%)": rel_diff * 100,
                    "Threshold": threshold * 100,
                    "Triggered Change Point": triggered,
                    "Initial Flag": flag_initial,
                    "Corrected Flag": flag_corrected
                })
            bead_df = pd.DataFrame(records)
            detailed_inspection.append(bead_df)

    detailed_windows_df = pd.concat(detailed_inspection, ignore_index=True)

    st.subheader("Raw and Smoothed/Masked Signal with Change Points")
    st.plotly_chart(raw_fig, use_container_width=True)

    st.subheader("Score Trace with Threshold")
    score_fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="orange",
        annotation_text="Threshold",
        annotation_position="top left"
    )
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Detailed Per-Window Change Point Analysis with Transparent Flags")
    st.dataframe(detailed_windows_df)
