import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
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

# --- Change Point Analysis ---
def analyze_change_points(signal: pd.Series, window_size: int, step_size: int, metric: str, threshold: float, mode: str):
    signal = signal.dropna().reset_index(drop=True)
    change_points = []
    diff_scores = []
    positions = []

    for start in range(0, len(signal) - 2 * window_size + 1, step_size):
        curr_window = signal[start:start + window_size]
        next_window = signal[start + window_size:start + 2 * window_size]

        if metric == "Mean":
            curr_stat = curr_window.mean()
            next_stat = next_window.mean()
        elif metric == "Median":
            curr_stat = curr_window.median()
            next_stat = next_window.median()
        elif metric == "Standard Deviation":
            curr_stat = curr_window.std()
            next_stat = next_window.std()
        else:
            raise ValueError("Invalid metric")

        if mode == "Absolute":
            diff = abs(curr_stat - next_stat)
        else:  # Relative
            denom = max(abs(curr_stat), 1e-6)
            diff = abs(curr_stat - next_stat) / denom

        diff_scores.append(diff)
        positions.append(start + window_size)

        if diff > threshold:
            change_points.append((start, start + 2 * window_size - 1, diff))

    return {
        "positions": positions,
        "diff_scores": diff_scores,
        "change_points": change_points
    }

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Change Point Detector v1")

st.sidebar.header("Upload Data")
test_zip = st.sidebar.file_uploader("ZIP file to Analyze", type="zip")

if test_zip:
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Select column for segmentation", columns)
    threshold_seg = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Select signal column for change point analysis", columns)

    if st.sidebar.button("Segment Beads"):
        with open("test.zip", "wb") as f:
            f.write(test_zip.getbuffer())

        test_files = extract_zip("test.zip", "test_data")

        def process_files(files):
            bead_data = defaultdict(list)
            for file in files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold_seg)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    bead_data[bead_num].append((os.path.basename(file), signal))
            return bead_data

        test_beads = process_files(test_files)
        st.session_state["test_beads"] = test_beads
        st.session_state["analysis_ready"] = True
        st.success("✅ Bead segmentation completed.")

if "test_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    test_beads = st.session_state["test_beads"]

    st.sidebar.header("Change Point Detection Settings")
    window_size = st.sidebar.number_input("Window Size (points)", min_value=10, value=100, step=10)
    step_size = st.sidebar.number_input("Step Size (points)", min_value=1, value=20, step=1)
    metric = st.sidebar.selectbox("Change Point Metric", ["Mean", "Median", "Standard Deviation"])
    threshold_mode = st.sidebar.selectbox("Threshold Mode", ["Absolute", "Relative (%)"])
    threshold_label = "Change Magnitude Threshold" if threshold_mode == "Absolute" else "Change Magnitude Threshold (%)"
    threshold = float(st.sidebar.text_input(threshold_label, value="0.10000"))
    if threshold_mode == "Relative (%)":
        threshold /= 100.0

    use_smoothing = st.sidebar.checkbox("Apply Signal Smoothing", value=False)
if use_smoothing:
    smoothing_method = st.sidebar.selectbox("Smoothing Method", ["Savitzky-Golay"])
    window_length = st.sidebar.number_input("Smoothing Window Length (odd number)", min_value=3, value=11, step=2)
    polyorder = st.sidebar.number_input("Polynomial Order", min_value=1, value=2, step=1)

selected_bead = st.selectbox("Select Bead Number to Display", sorted(test_beads.keys()))

    fig = go.Figure()
    score_fig = go.Figure()
    final_summary = []
    global_summary_dict = defaultdict(lambda: {"Change Points": []})

    for bead_num in sorted(test_beads.keys()):
        for fname, raw_signal in test_beads[bead_num]:
            signal = raw_signal.copy()
            if use_smoothing and smoothing_method == "Savitzky-Golay" and len(signal) >= window_length:
                signal = pd.Series(savgol_filter(signal, window_length, polyorder))
            result = analyze_change_points(signal, window_size, step_size, metric, threshold, threshold_mode)
            color = 'red' if result["change_points"] else 'black'

            if bead_num == selected_bead:
                fig.add_trace(go.Scatter(
                    y=signal,
                    mode='lines',
                    name=fname,
                    line=dict(color=color, width=1.5)
                ))
                for start, end, diff in result["change_points"]:
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=min(signal), y1=max(signal),
                        fillcolor="rgba(255,0,0,0.2)",
                        line=dict(width=0),
                        layer="below"
                    )
                score_fig.add_trace(go.Scatter(
                    x=result["positions"],
                    y=result["diff_scores"],
                    mode="lines+markers",
                    name=fname
                ))
                score_fig.add_trace(go.Scatter(
                    x=result["positions"],
                    y=[threshold]*len(result["positions"]),
                    mode="lines",
                    name="Threshold",
                    line=dict(color="orange", dash="dash")
                ))

            global_summary_dict[fname]["Change Points"].extend(
                [(bead_num, start, end, round(diff, 4)) for start, end, diff in result["change_points"]]
            )

            if bead_num == selected_bead:
                final_summary.append({
                    "File Name": fname,
                    "Bead Number": bead_num,
                    "Total Change Points": len(result["change_points"]),
                    "Result": "NOK" if result["change_points"] else "OK"
                })

                with st.expander(f"Change Point Details for {fname} - Bead {bead_num}"):
                    df_window = pd.DataFrame(result["change_points"], columns=["Start", "End", f"{metric} Diff"])
                    df_window[f"{metric} Diff > Threshold"] = df_window[f"{metric} Diff"].apply(lambda x: x > threshold)
                    st.dataframe(df_window)

    st.plotly_chart(fig, use_container_width=True, key="signal_plot")

    if use_smoothing:
        st.markdown("### Smoothed Signal Plot")
        fig_smooth = go.Figure()
        for fname, raw_signal in test_beads[selected_bead]:
            if smoothing_method == "Savitzky-Golay" and len(raw_signal) >= window_length:
                smooth_signal = savgol_filter(raw_signal, window_length, polyorder)
                fig_smooth.add_trace(go.Scatter(
                    y=smooth_signal,
                    mode='lines',
                    name=f"{fname} (Smoothed)"
                ))
        st.plotly_chart(fig_smooth, use_container_width=True, key="smoothed_signal_plot")
    st.markdown("### Change Magnitude Score Trace (Per Window)")
    st.plotly_chart(score_fig, use_container_width=True, key="main_score_trace")

    st.markdown("### Comparison: Absolute and Relative Score Traces")
    abs_score_fig = go.Figure()
    rel_score_fig = go.Figure()

    for bead_num in sorted(test_beads.keys()):
        for fname, signal in test_beads[bead_num]:
            result_abs = analyze_change_points(signal, window_size, step_size, metric, threshold, mode="Absolute")
            result_rel = analyze_change_points(signal, window_size, step_size, metric, threshold, mode="Relative")

            abs_score_fig.add_trace(go.Scatter(
                x=result_abs["positions"],
                y=result_abs["diff_scores"],
                mode="lines+markers",
                name=f"{fname} (Abs)"
            ))
            abs_score_fig.add_trace(go.Scatter(
                x=result_abs["positions"],
                y=[threshold]*len(result_abs["positions"]),
                mode="lines",
                name="Absolute Threshold",
                line=dict(color="orange", dash="dash")
            ))

            rel_score_fig.add_trace(go.Scatter(
                x=result_rel["positions"],
                y=[v * 100 for v in result_rel["diff_scores"]],
                mode="lines+markers",
                name=f"{fname} (Rel %)")
            )
            rel_score_fig.add_trace(go.Scatter(
                x=result_rel["positions"],
                y=[threshold * 100]*len(result_rel["positions"]),
                mode="lines",
                name="Relative Threshold (%)",
                line=dict(color="green", dash="dot")
            ))

    st.markdown("#### Absolute Mode Scores")
    st.plotly_chart(abs_score_fig, use_container_width=True, key="abs_trace")

    st.markdown("#### Relative Mode Scores (%)")
    st.plotly_chart(rel_score_fig, use_container_width=True, key="rel_trace")

    st.markdown("### Change Point Summary Table (Selected Bead)")
    st.dataframe(pd.DataFrame(final_summary))

    global_summary = []
    for fname, info in global_summary_dict.items():
        nok_beads = sorted(set(bead for bead, _, _, _ in info["Change Points"]))
        global_summary.append({
            "File Name": fname,
            "NOK Beads": ", ".join(map(str, nok_beads)),
            "Total Change Points": len(info["Change Points"]),
            "Welding Result": "NOK" if nok_beads else "OK"
        })

    st.markdown("### Global Change Point Summary")
    st.dataframe(pd.DataFrame(global_summary))
