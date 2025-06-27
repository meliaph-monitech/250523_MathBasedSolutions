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
        else:
            raise ValueError("Invalid metric")

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

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Change Point Detector (Aluminum/Copper Auto Classification")

st.sidebar.header("Upload and Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()

    seg_col = st.sidebar.selectbox("Column for Segmentation", columns, key="seg_col")
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=3.0, key="seg_thresh")
    signal_col = st.sidebar.selectbox("Signal Column for Analysis", columns, key="sig_col")

    analysis_percent = st.sidebar.slider("% of Signal Length to Consider for NOK Decision", min_value=10, max_value=100, value=50, step=10)

    if st.sidebar.button("Segment Beads"):
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        file_paths = extract_zip("uploaded.zip", "data")

        raw_beads = defaultdict(list)
        bead_lengths = []
        for path in file_paths:
            df = pd.read_csv(path)
            beads = segment_beads(df, seg_col, seg_thresh)
            for bead_num, (start, end) in enumerate(beads, 1):
                segment = df.iloc[start:end+1][signal_col].reset_index(drop=True)
                raw_beads[bead_num].append((os.path.basename(path), segment))
                bead_lengths.append(len(segment))

        median_length = np.median(bead_lengths)
        st.session_state["raw_beads"] = raw_beads
        st.session_state["analysis_ready"] = True
        st.session_state["analysis_percent"] = analysis_percent
        st.session_state["median_length"] = median_length
        st.success(f"✅ Bead segmentation completed. Median length: {median_length}")

if "raw_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    raw_beads = st.session_state["raw_beads"]
    analysis_percent = st.session_state["analysis_percent"]
    median_length = st.session_state["median_length"]

    st.sidebar.header("Aluminum Specific Settings")
    alu_ignore_thresh = st.sidebar.number_input("Aluminum Ignore Threshold (Filter Values Above)", value=5000.0)

    st.sidebar.header("Smoothing & Detection")
    use_smooth = st.sidebar.checkbox("Apply Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 199, 199, step=2)
        poly = st.sidebar.number_input("Polynomial Order", 1, 5, 5)

    win_size = st.sidebar.number_input("Window Size (Analysis)", 10, 1000, 350, step=10)
    step = st.sidebar.number_input("Step Size", 1, 500, 175)
    metric = st.sidebar.selectbox("Metric", ["Mean", "Median", "Standard Deviation"])
    mode = st.sidebar.selectbox("Threshold Mode", ["Absolute", "Relative (%)"])
    thresh_input = st.sidebar.text_input("Change Magnitude Threshold", value="15")
    thresh = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Raw and Smoothed Signal for Bead {selected_bead}")
    raw_plot = go.Figure()
    score_plot = go.Figure()
    abs_plot = go.Figure()
    rel_plot = go.Figure()

    table_rows = []
    global_summary = defaultdict(list)

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= 0.7 * median_length else "Copper"
            sig = raw_sig
            if bead_type == "Aluminum":
                sig = sig[sig < alu_ignore_thresh]

            if use_smooth and len(sig) >= win_size:
                sig = pd.Series(savgol_filter(sig, win_len, poly))

            result = analyze_change_points(sig, win_size, step, metric, thresh, mode)
            color = 'red' if result['change_points'] else 'black'

            if bead_num == selected_bead:
                raw_plot.add_trace(go.Scatter(y=sig, name=f"{fname} ({bead_type})", mode='lines', line=dict(color=color)))
                for start, end, _ in result["change_points"]:
                    raw_plot.add_shape(type="rect", x0=start, x1=end, y0=min(sig), y1=max(sig), fillcolor="rgba(255,0,0,0.2)", line_width=0)

                score_plot.add_trace(go.Scatter(x=result["positions"], y=result["abs_scores"] if mode == "Absolute" else result["rel_scores"], mode='lines+markers', name=fname))
                score_plot.add_trace(go.Scatter(x=result["positions"], y=[thresh]*len(result["positions"]), mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))
                abs_plot.add_trace(go.Scatter(x=result["positions"], y=result["abs_scores"], name=f"{fname} Abs", mode='lines'))
                rel_plot.add_trace(go.Scatter(x=result["positions"], y=[v * 100 for v in result["rel_scores"]], name=f"{fname} Rel %", mode='lines'))

            max_index = int(len(raw_sig) * (analysis_percent / 100))
            nok = any(end < max_index for (start, end, _) in result["change_points"])
            if nok:
                global_summary[fname].append(f"{bead_num} ({bead_type})")

            table_rows.append({
                "File": fname,
                "Bead": bead_num,
                "Bead Type": bead_type,
                "Change Points": len(result["change_points"]),
                "Result": "NOK" if nok else "OK"
            })

    st.plotly_chart(raw_plot, use_container_width=True)
    st.subheader("Score Trace (Change Magnitude per Window)")
    st.plotly_chart(score_plot, use_container_width=True)

    st.subheader("Comparison: Absolute and Relative Score Traces")
    st.plotly_chart(abs_plot, use_container_width=True)
    st.plotly_chart(rel_plot, use_container_width=True)

    st.subheader("Change Point Summary (Selected Bead)")
    st.dataframe(pd.DataFrame([row for row in table_rows if row["Bead"] == selected_bead]))

    st.subheader("Global Summary Table")
    global_table = pd.DataFrame([{ "File": fname, "NOK Beads": ", ".join(beads), "Welding Result": "NOK" if beads else "OK" } for fname, beads in global_summary.items()])
    st.dataframe(global_table)
