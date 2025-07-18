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
st.title("Detailed Change Point Inspection per Bead")

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

    corrected_summary = defaultdict(lambda: {"Corrected NOK": [], "Corrected OK_Check": []})

    bead_flags_per_file = defaultdict(lambda: defaultdict(str))

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

            flag_corrected = flag_initial

            if flag_initial in ["NOK", "OK_Check"]:
                clip_threshold = np.percentile(raw_sig, 75)
                filtered_array = np.where(raw_sig > clip_threshold, clip_threshold, raw_sig)
                if use_smooth and len(filtered_array) >= win_len:
                    filtered_array = savgol_filter(filtered_array, win_len, polyorder)
                filtered_sig = pd.Series(filtered_array)

                filtered_result = analyze_change_points(filtered_sig, win_size, step_size, metric, threshold)
                cp_in_region_filtered = [cp for cp in filtered_result["change_points"] if cp[1] < nok_region_limit]

                has_nok = any(cp[2] > threshold for cp in cp_in_region_filtered)
                has_ok_check = any(cp[2] < -threshold for cp in cp_in_region_filtered)

                if flag_initial == "NOK" and not has_nok:
                    flag_corrected = "OK"
                elif flag_initial == "OK_Check" and not has_ok_check:
                    flag_corrected = "OK"

            if flag_corrected == "NOK":
                corrected_summary[fname]["Corrected NOK"].append(f"{bead_num} ({bead_type})")
            elif flag_corrected == "OK_Check":
                corrected_summary[fname]["Corrected OK_Check"].append(f"{bead_num} ({bead_type})")

            bead_flags_per_file[fname][bead_num] = flag_corrected

    st.subheader("Corrected NOK and OK_Check Beads Summary Across All Beads")
    if corrected_summary:
        corrected_table = pd.DataFrame([
            {
                "File": file,
                "Corrected NOK": ", ".join(beads["Corrected NOK"]) if beads["Corrected NOK"] else "-",
                "Corrected OK_Check": ", ".join(beads["Corrected OK_Check"]) if beads["Corrected OK_Check"] else "-"
            }
            for file, beads in corrected_summary.items()
        ])
        st.dataframe(corrected_table)
    else:
        st.write("✅ No NOK or OK_Check beads detected across all files and beads after recheck.")

    with st.sidebar:
        selected_bead = st.selectbox("Select Bead Number for Detailed Inspection", bead_options)

    raw_fig = go.Figure()
    score_fig = go.Figure()
    detailed_inspection = []

    for bead_num in [selected_bead]:
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

            flag = bead_flags_per_file[fname][bead_num]
            color = 'red' if flag == "NOK" else 'blue' if flag == "OK_Check" else 'black'

            if flag in ["NOK", "OK_Check"]:
                clip_threshold = np.percentile(raw_sig, 75)
                filtered_array = np.where(raw_sig > clip_threshold, clip_threshold, raw_sig)
                if use_smooth and len(filtered_array) >= win_len:
                    filtered_array = savgol_filter(filtered_array, win_len, polyorder)
                plot_sig = pd.Series(filtered_array)
            else:
                plot_sig = sig

            raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)"))
            raw_fig.add_trace(go.Scatter(y=plot_sig, mode='lines', name=f"{fname} (filtered)", line=dict(color=color)))

            for start, end, _ in result["change_points"]:
                raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, layer="below")

            y_scores = [v * 100 for v in result["rel_scores"]]
            score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Rel Diff (%)"))

            signal_clean = sig.reset_index(drop=True)
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
                    "Flag": flag
                })
            bead_df = pd.DataFrame(records)
            detailed_inspection.append(bead_df)

    detailed_windows_df = pd.concat(detailed_inspection, ignore_index=True)

    st.subheader("Raw and Smoothed Signal with Change Points")
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

    st.subheader("Detailed Per-Window Change Point Analysis")
    st.dataframe(detailed_windows_df)
