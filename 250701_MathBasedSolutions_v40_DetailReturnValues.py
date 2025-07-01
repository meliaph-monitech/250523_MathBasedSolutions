import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

with st.sidebar:
    uploaded_zip = st.file_uploader("Upload ZIP of CSVs", type="zip")

# if uploaded_zip:
#     with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
#         first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
#         with zip_ref.open(first_csv) as f:
#             sample_df = pd.read_csv(f)

#     # Hard-coded parameters as requested
#     seg_col = sample_df.columns[2]
#     signal_col = sample_df.columns[0]
#     seg_thresh = 3.0
#     analysis_percent = 100
#     alu_ignore_thresh = 3.0
#     cu_ignore_thresh = 3.0
#     use_smooth = True
#     win_len = 199
#     polyorder = 5
#     win_size = 350
#     step_size = 175
#     metric = "Median"
#     mode = "Relative (%)"
#     thresh_input = "15"
#     threshold = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)
if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)

    use_smooth = True
    metric = "Median"
    mode = "Relative (%)"
    seg_col = sample_df.columns[2]
    seg_thresh = 3.0
    
    with st.sidebar: 
        signal_col = st.selectbox("Column for Signal Analysis (signal_col)", sample_df.columns, index=0)
        analysis_percent = st.number_input("Analysis Percent (analysis_percent)", min_value=1, max_value=100, value=100, step=1)
        alu_ignore_thresh = st.number_input("Aluminum Ignore Threshold (alu_ignore_thresh)", value=3.0, step=0.1)
        cu_ignore_thresh = st.number_input("Copper Ignore Threshold (cu_ignore_thresh)", value=3.0, step=0.1)
        win_len = st.number_input("Smoothing Window Length (win_len, odd)", min_value=3, max_value=999, value=199, step=2)
        polyorder = st.number_input("Smoothing Polynomial Order (polyorder)", min_value=1, max_value=5, value=5, step=1)
        win_size = st.number_input("Window Size (win_size)", min_value=10, max_value=5000, value=350, step=10)
        step_size = st.number_input("Step Size (step_size)", min_value=1, max_value=1000, value=175, step=5)
        thresh_input = st.text_input("Change Magnitude Threshold (%)", "15")
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

    # --- GLOBAL NOK/NOK_CHECK SUMMARY ACROSS ALL BEADS ---
    global_summary = defaultdict(list)

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            sig = raw_sig.copy()
            if bead_type == "Aluminum":
                sig = np.minimum(sig, alu_ignore_thresh)
            else:
                sig = np.minimum(sig, cu_ignore_thresh)
            if use_smooth and len(sig) >= win_size:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            result = analyze_change_points(sig, win_size, step_size, metric, threshold, mode)
            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)
            cp_in_region = [cp for cp in result["change_points"] if cp[1] < nok_region_limit]
            flag = "OK" if len(cp_in_region) == 0 else ("NOK" if len(cp_in_region) == 1 else "NOK_Check")
            if flag in ["NOK", "NOK_Check"]:
                global_summary[fname].append(f"{bead_num} ({bead_type})")

    st.subheader("Global NOK and NOK_Check Beads Summary Across All Beads")
    if global_summary:
        global_table = pd.DataFrame([
            {"File": file, "NOK/NOK_Check Beads": ", ".join(beads)}
            for file, beads in global_summary.items()
        ])
        st.dataframe(global_table)
    else:
        st.write("✅ No NOK or NOK_Check beads detected across all files and all beads.")

    with st.sidebar:
        selected_bead = st.selectbox("Select Bead Number for Detailed Inspection", bead_options)

    raw_fig = go.Figure()
    score_fig = go.Figure()
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

            result = analyze_change_points(sig, win_size, step_size, metric, threshold, mode)
            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)
            cp_in_region = [cp for cp in result["change_points"] if cp[1] < nok_region_limit]
            flag = "OK" if len(cp_in_region) == 0 else ("NOK" if len(cp_in_region) == 1 else "NOK_Check")

            for start, end, _ in result["change_points"]:
                raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, layer="below", line_width=0)

            raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)"))
            color = 'red' if flag != "OK" else 'black'
            raw_fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} (filtered)", line=dict(color=color)))

            y_scores = result["abs_scores"] if mode == "Absolute" else [v*100 for v in result["rel_scores"]]
            score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Score"))
            # score_fig.add_trace(go.Scatter(
            #     x=result["positions"],
            #     y=[threshold*100 if mode=="Relative (%)" else threshold]*len(result["positions"]),
            #     mode='lines',
            #     name="Threshold",
            #     line=dict(color="orange", dash="dash")
            # ))

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
        y=threshold*100 if mode == "Relative (%)" else threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text="Threshold",
        annotation_position="top left"
    )
    st.plotly_chart(score_fig, use_container_width=True)


    # df_vis = detailed_windows_df.copy()
    # df_vis['Color'] = np.where(df_vis['Triggered Change Point'], 'Triggered', 'Not Triggered')

    # scatter_fig = go.Figure()

    # # Triggered
    # triggered = df_vis[df_vis['Triggered Change Point'] == True]
    # scatter_fig.add_trace(go.Scatter(
    #     x=triggered["Start Index"],
    #     y=triggered["Rel Diff (%)"] if mode == "Relative (%)" else triggered["Abs Diff"],
    #     mode='markers',
    #     marker=dict(color='red'),
    #     name='Triggered'
    # ))

    # # Not Triggered
    # not_triggered = df_vis[df_vis['Triggered Change Point'] == False]
    # scatter_fig.add_trace(go.Scatter(
    #     x=not_triggered["Start Index"],
    #     y=not_triggered["Rel Diff (%)"] if mode == "Relative (%)" else not_triggered["Abs Diff"],
    #     mode='markers',
    #     marker=dict(color='black'),
    #     name='Not Triggered'
    # ))
    
    # scatter_fig.add_hline(
    #     y=threshold*100 if mode=="Relative (%)" else threshold,
    #     line_dash="dash",
    #     line_color="orange",
    #     annotation_text="Threshold",
    #     annotation_position="top left"
    # )

    # scatter_fig.update_layout(
    #     title=f"Window-based Change Detection for Bead {selected_bead}",
    #     xaxis_title="Start Index",
    #     yaxis_title="Rel Diff (%)" if mode == "Relative (%)" else "Abs Diff"
    # )

    # st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Detailed Per-Window Change Point Analysis")
    st.dataframe(detailed_windows_df)
