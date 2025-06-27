# New clean version will be constructed here based on clarified requirements.

import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter

# --- Utility: File Extraction ---
def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

# --- Utility: Bead Segmentation ---
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

# --- Core: Change Point Detection ---
def analyze_change_points(signal, window_size, step_size, metric, threshold, mode):
    signal = signal.dropna().reset_index(drop=True)
    change_points = []
    diff_scores = []
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
        else:
            raise ValueError("Invalid metric")

        if mode == "Absolute":
            diff = abs(v1 - v2)
        else:
            diff = abs(v1 - v2) / max(abs(v1), 1e-6)

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
st.title("Change Point Detector (with Optional Smoothing)")

st.sidebar.header("Upload and Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()

    seg_col = st.sidebar.selectbox("Column for Segmentation", columns, key="seg_col")
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=0.0, key="seg_thresh")
    signal_col = st.sidebar.selectbox("Signal Column for Analysis", columns, key="sig_col")

    if st.sidebar.button("Segment Beads"):
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        file_paths = extract_zip("uploaded.zip", "data")

        raw_beads = defaultdict(list)
        for path in file_paths:
            df = pd.read_csv(path)
            beads = segment_beads(df, seg_col, seg_thresh)
            for bead_num, (start, end) in enumerate(beads, 1):
                segment = df.iloc[start:end+1][signal_col].reset_index(drop=True)
                raw_beads[bead_num].append((os.path.basename(path), segment))

        st.session_state["raw_beads"] = raw_beads
        st.session_state["analysis_ready"] = True
        st.success("✅ Bead segmentation completed.")

if "raw_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    raw_beads = st.session_state["raw_beads"]

    st.sidebar.header("Smoothing & Detection")
    use_smooth = st.sidebar.checkbox("Apply Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 199, 11, step=2)
        poly = st.sidebar.number_input("Polynomial Order", 1, 5, 2)

    win_size = st.sidebar.number_input("Window Size (Analysis)", 10, 1000, 100, step=10)
    step = st.sidebar.number_input("Step Size", 1, 500, 20)
    metric = st.sidebar.selectbox("Metric", ["Mean", "Median", "Standard Deviation"])
    mode = st.sidebar.selectbox("Threshold Mode", ["Absolute", "Relative (%)"])
    thresh_input = st.sidebar.text_input("Change Magnitude Threshold", value="0.1")
    thresh = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Raw and Smoothed Signal for Bead {selected_bead}")
    raw_plot = go.Figure()
    score_plot = go.Figure()
    table_rows = []
    global_summary = defaultdict(list)

    for fname, signal in raw_beads[selected_bead]:
        raw_plot.add_trace(go.Scatter(y=signal, name=fname, mode='lines', line=dict(width=1)))

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            sig = raw_sig
            if use_smooth and len(sig) >= win_len:
                sig = pd.Series(savgol_filter(sig, win_len, poly))

            result = analyze_change_points(sig, win_size, step, metric, thresh, mode)
            color = 'red' if result['change_points'] else 'black'

            if bead_num == selected_bead:
                raw_plot.add_trace(go.Scatter(y=sig, name=f"{fname} (smoothed)" if use_smooth else fname, mode='lines', line=dict(color=color)))
                for start, end, _ in result["change_points"]:
                    raw_plot.add_shape(type="rect", x0=start, x1=end, y0=min(sig), y1=max(sig), fillcolor="rgba(255,0,0,0.2)", line_width=0)
                score_plot.add_trace(go.Scatter(x=result["positions"], y=result["diff_scores"], mode='lines+markers', name=fname))
                score_plot.add_trace(go.Scatter(x=result["positions"], y=[thresh]*len(result["positions"]), mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))

            if result["change_points"]:
                global_summary[fname].append(str(bead_num))
            table_rows.append({
                "File": fname,
                "Bead": bead_num,
                "Change Points": len(result["change_points"]),
                "Result": "NOK" if result["change_points"] else "OK"
            })

    st.plotly_chart(raw_plot, use_container_width=True)
    st.subheader("Score Trace (Change Magnitude per Window)")
    st.plotly_chart(score_plot, use_container_width=True)

    st.subheader("Change Point Summary (Selected Bead)")
    st.dataframe(pd.DataFrame([row for row in table_rows if row["Bead"] == selected_bead]))

    st.subheader("Global Summary Table")
    global_table = pd.DataFrame([{ "File": fname, "NOK Beads": ", ".join(beads), "Welding Result": "NOK" if beads else "OK" } for fname, beads in global_summary.items()])
    st.dataframe(global_table)
