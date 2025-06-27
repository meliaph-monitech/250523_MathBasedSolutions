import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter
import shutil

def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            path = os.path.join(extract_dir, file)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

# # --- Utility: File Extraction ---
# def extract_zip(uploaded_file, extract_dir):
#     if os.path.exists(extract_dir):
#         for file in os.listdir(extract_dir):
#             os.remove(os.path.join(extract_dir, file))
#     else:
#         os.makedirs(extract_dir)

#     with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
#         zip_ref.extractall(extract_dir)

#     return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

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
st.title("Change Point Detector with Aluminum/Copper Detection")

st.sidebar.header("Upload and Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()

    seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=3.0)
    signal_col = st.sidebar.selectbox("Signal Column for Analysis", columns)
    analysis_percent = st.sidebar.slider("% of Signal Length to Consider for NOK Decision", 10, 100, 50, 10)

    if st.sidebar.button("Segment Beads"):
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        files = extract_zip("uploaded.zip", "data")

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

        st.session_state["raw_beads"] = raw_beads
        st.session_state["analysis_ready"] = True
        st.session_state["split_length"] = split_length
        st.session_state["analysis_percent"] = analysis_percent
        st.success(f"✅ Bead segmentation completed. Split length for Al/Cu: {split_length}")

if "raw_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    raw_beads = st.session_state["raw_beads"]
    split_length = st.session_state["split_length"]
    analysis_percent = st.session_state["analysis_percent"]

    st.sidebar.header("Aluminum Filtering")
    alu_ignore_thresh = st.sidebar.number_input("Aluminum Ignore Threshold (Filter Above)", value=5000.0)

    st.sidebar.header("Smoothing & Detection")
    use_smooth = st.sidebar.checkbox("Apply Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 399, 199, step=2)
        polyorder = st.sidebar.number_input("Polynomial Order", 1, 5, 5)

    win_size = st.sidebar.number_input("Window Size (Analysis)", 10, 1000, 350, 10)
    step_size = st.sidebar.number_input("Step Size", 1, 500, 175)
    metric = st.sidebar.selectbox("Metric", ["Mean", "Median", "Standard Deviation"])
    mode = st.sidebar.selectbox("Threshold Mode", ["Absolute", "Relative (%)"])
    thresh_input = st.sidebar.text_input("Change Magnitude Threshold", "15")
    threshold = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Raw and Smoothed Signal for Bead {selected_bead}")
    raw_fig = go.Figure()
    score_fig = go.Figure()

    table_data = []
    global_summary = defaultdict(list)

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            sig = raw_sig.copy()
            if bead_type == "Aluminum":
                sig = sig[sig < alu_ignore_thresh]
            if use_smooth and len(sig) >= win_size:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            result = analyze_change_points(sig, win_size, step_size, metric, threshold, mode)
            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)
            nok = any(end < nok_region_limit for start, end, _ in result["change_points"])
            if nok:
                global_summary[fname].append(f"{bead_num} ({bead_type})")

            table_data.append({
                "File": fname,
                "Bead": bead_num,
                "Bead Type": bead_type,
                "Change Points": len(result["change_points"]),
                "Result": "NOK" if nok else "OK"
            })

            if bead_num == selected_bead:
                # Add shaded areas to BACK
                for start, end, _ in result["change_points"]:
                    raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.05, layer="below", line_width=0)

                raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)", line=dict(width=1)))
                color = 'red' if result["change_points"] else 'black'
                raw_fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} (filtered)", line=dict(color=color)))

                y_scores = result["abs_scores"] if mode == "Absolute" else [v*100 for v in result["rel_scores"]]
                score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Score"))
                score_fig.add_trace(go.Scatter(x=result["positions"], y=[threshold*100 if mode=="Relative (%)" else threshold]*len(result["positions"]), mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))

    st.plotly_chart(raw_fig, use_container_width=True)
    st.subheader("Score Trace (Change Magnitude per Window)")
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Change Point Summary Table")
    st.dataframe(pd.DataFrame(table_data))

    st.subheader("Global NOK Beads Summary")
    global_table = pd.DataFrame([{ "File": k, "NOK Beads": ", ".join(v) } for k, v in global_summary.items()])
    st.dataframe(global_table)
