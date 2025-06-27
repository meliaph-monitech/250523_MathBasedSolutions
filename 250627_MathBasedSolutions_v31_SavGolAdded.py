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

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Change Point Detector with Optional Smoothing")

st.sidebar.header("Upload & Segmentation Settings")
test_zip = st.sidebar.file_uploader("Upload ZIP file of CSVs", type="zip")

if test_zip:
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Segmentation Column", columns, key="seg_col")
    threshold_seg = st.sidebar.number_input("Segmentation Threshold", value=0.0, key="seg_threshold")
    signal_column = st.sidebar.selectbox("Signal Column for Analysis", columns, key="sig_col")

    st.sidebar.header("Optional Smoothing")
    use_smoothing = st.sidebar.checkbox("Apply Smoothing", value=False, key="smoothing_toggle")
    if use_smoothing:
        window_length = st.sidebar.number_input("Window Length (odd)", min_value=3, value=11, step=2, key="smooth_win")
        polyorder = st.sidebar.number_input("Polynomial Order", min_value=1, value=2, step=1, key="smooth_poly")

    st.sidebar.header("Change Point Detection Settings")
    window_size = st.sidebar.number_input("Window Size", min_value=10, value=100, step=10, key="win_size")
    step_size = st.sidebar.number_input("Step Size", min_value=1, value=20, step=1, key="step_size")
    metric = st.sidebar.selectbox("Metric", ["Mean", "Median", "Standard Deviation"], key="metric")
    threshold_mode = st.sidebar.selectbox("Threshold Mode", ["Absolute", "Relative (%)"], key="mode")
    threshold_label = "Change Magnitude Threshold" if threshold_mode == "Absolute" else "Change Magnitude Threshold (%)"
    threshold = float(st.sidebar.text_input(threshold_label, value="0.1", key="threshold_val"))
    if threshold_mode == "Relative (%)":
        threshold /= 100.0

    st.sidebar.markdown("""
Use the plots below to understand the score distributions for your data. Adjust the threshold accordingly:
- For **Absolute**, observe the actual value range.
- For **Relative**, aim for a small % (e.g., 3–10%).
""")

    if st.sidebar.button("Segment Beads and Run Analysis"):
        with open("uploaded.zip", "wb") as f:
            f.write(test_zip.getbuffer())
        test_files = extract_zip("uploaded.zip", "data")

        raw_beads = defaultdict(list)
        smoothed_beads = defaultdict(list)

        for file in test_files:
            df = pd.read_csv(file)
            segments = segment_beads(df, filter_column, threshold_seg)
            for bead_num, (start, end) in enumerate(segments, 1):
                raw_signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                raw_beads[bead_num].append((os.path.basename(file), raw_signal))
                if use_smoothing and len(raw_signal) >= window_length:
                    smoothed = pd.Series(savgol_filter(raw_signal, window_length, polyorder))
                else:
                    smoothed = raw_signal
                smoothed_beads[bead_num].append((os.path.basename(file), smoothed))

        selected_bead = st.selectbox("Select Bead Number", sorted(smoothed_beads.keys()))

        st.subheader("Raw and Smoothed Signal")
        fig_raw = go.Figure()
        fig_smooth = go.Figure()
        for fname, raw_sig in raw_beads[selected_bead]:
            fig_raw.add_trace(go.Scatter(y=raw_sig, mode='lines', name=fname))
        for fname, smooth_sig in smoothed_beads[selected_bead]:
            fig_smooth.add_trace(go.Scatter(y=smooth_sig, mode='lines', name=fname))
        st.plotly_chart(fig_raw, use_container_width=True)
        if use_smoothing:
            st.plotly_chart(fig_smooth, use_container_width=True)

        st.subheader("Change Magnitude Score Trace")
        score_fig = go.Figure()
        final_summary = []
        global_summary_dict = defaultdict(list)

        for bead_num in sorted(smoothed_beads.keys()):
            for (fname, signal) in smoothed_beads[bead_num]:
                result = analyze_change_points(signal, window_size, step_size, metric, threshold, threshold_mode)
                nok = bool(result['change_points'])
                if bead_num == selected_bead:
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=result["diff_scores"], mode='lines+markers', name=fname))
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=[threshold]*len(result["positions"]), mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))
                if nok:
                    global_summary_dict[fname].append(str(bead_num))
                final_summary.append({
                    "File Name": fname,
                    "Bead Number": bead_num,
                    "Total Change Points": len(result["change_points"]),
                    "Result": "NOK" if nok else "OK"
                })

        st.plotly_chart(score_fig, use_container_width=True)
        st.subheader("Final Summary Table")
        st.dataframe(pd.DataFrame(final_summary))

        st.subheader("Global NOK Bead Summary")
        global_summary = [{"File Name": fname, "NOK Beads": ", ".join(sorted(beads)), "Welding Result": "NOK" if beads else "OK"} for fname, beads in global_summary_dict.items()]
        st.dataframe(pd.DataFrame(global_summary))
