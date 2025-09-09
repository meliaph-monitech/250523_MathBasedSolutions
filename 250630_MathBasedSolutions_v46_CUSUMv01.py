import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.signal import savgol_filter
import shutil

# --- Utility: File Extraction ---
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

# --- CUSUM Change Point Detection ---
def cusum_change_points(signal, threshold, drift=None):
    """
    CUSUM-based change point detection
    
    Args:
        signal: pandas Series or numpy array
        threshold: Detection threshold (higher = less sensitive)
        drift: Optional drift parameter (default: threshold/2)
    
    Returns:
        Dictionary with change points and CUSUM scores
    """
    signal = pd.Series(signal).dropna().reset_index(drop=True)
    
    if len(signal) < 10:  # Too short for meaningful analysis
        return {"positions": [], "cusum_pos": [], "cusum_neg": [], "change_points": []}
    
    # Calculate signal statistics for drift estimation
    signal_mean = signal.mean()
    signal_std = signal.std()
    
    if drift is None:
        drift = threshold / 2
    
    # Initialize CUSUM variables
    cusum_pos = np.zeros(len(signal))
    cusum_neg = np.zeros(len(signal))
    change_points = []
    
    # Calculate CUSUM
    for i in range(1, len(signal)):
        # Positive CUSUM (detects upward shifts)
        cusum_pos[i] = max(0, cusum_pos[i-1] + (signal[i] - signal_mean) - drift)
        
        # Negative CUSUM (detects downward shifts)  
        cusum_neg[i] = max(0, cusum_neg[i-1] - (signal[i] - signal_mean) - drift)
        
        # Check for change points
        if cusum_pos[i] > threshold:
            change_points.append((i, 'upward', cusum_pos[i]))
            cusum_pos[i] = 0  # Reset after detection
            
        if cusum_neg[i] > threshold:
            change_points.append((i, 'downward', cusum_neg[i]))
            cusum_neg[i] = 0  # Reset after detection
    
    return {
        "positions": list(range(len(signal))),
        "cusum_pos": cusum_pos.tolist(),
        "cusum_neg": cusum_neg.tolist(),
        "change_points": [(cp[0], cp[0], cp[2]) for cp in change_points]  # Format: (start, end, score)
    }

# --- Core: Change Point Detection (Original Sliding Window) ---
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
st.title("Change Point Detector with CUSUM Algorithm Option")

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

    st.sidebar.header("Filtering")
    alu_ignore_thresh = st.sidebar.number_input("Aluminum Ignore Threshold (Filter Above)", value=3.0)
    cu_ignore_thresh = st.sidebar.number_input("Copper Ignore Threshold (Filter Above)", value=3.0)

    st.sidebar.header("Smoothing")
    use_smooth = st.sidebar.checkbox("Apply Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 499, 199, step=2)
        polyorder = st.sidebar.number_input("Polynomial Order", 1, 5, 5)

    # --- NEW: Algorithm Selection ---
    st.sidebar.header("Detection Algorithm")
    algorithm = st.sidebar.selectbox("Choose Algorithm", ["Sliding Window (Original)", "CUSUM"])
    
    if algorithm == "Sliding Window (Original)":
        st.sidebar.subheader("Sliding Window Parameters")
        win_size = st.sidebar.number_input("Window Size (Analysis)", 10, 1000, 350, 10)
        step_size = st.sidebar.number_input("Step Size", 1, 500, 175)
        metric = st.sidebar.selectbox("Metric", ["Median", "Mean", "Standard Deviation"])
        mode = st.sidebar.selectbox("Threshold Mode", ["Relative (%)", "Absolute"])
        thresh_input = st.sidebar.text_input("Change Magnitude Threshold", "15")
        threshold = float(thresh_input) / 100 if mode == "Relative (%)" else float(thresh_input)
    
    else:  # CUSUM
        st.sidebar.subheader("CUSUM Parameters")
        cusum_threshold = st.sidebar.number_input("CUSUM Threshold", 1.0, 100.0, 10.0, 0.5)
        cusum_drift = st.sidebar.number_input("Drift Parameter (0=auto)", 0.0, 50.0, 0.0, 0.1)
        if cusum_drift == 0.0:
            cusum_drift = None

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Signal Analysis for Bead {selected_bead} using {algorithm}")
    raw_fig = go.Figure()
    score_fig = go.Figure()

    table_data = []
    global_summary = defaultdict(list)

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            sig = raw_sig.copy()

            if bead_type == "Aluminum":
                sig = np.minimum(sig, alu_ignore_thresh)
            elif bead_type == "Copper":
                sig = np.minimum(sig, cu_ignore_thresh)

            if use_smooth and len(sig) >= (win_len if algorithm == "Sliding Window (Original)" else 10):
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            # Apply selected algorithm
            if algorithm == "Sliding Window (Original)":
                result = analyze_change_points(sig, win_size, step_size, metric, threshold, mode)
            else:  # CUSUM
                result = cusum_change_points(sig, cusum_threshold, cusum_drift)

            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)

            # Determine NOK, NOK_Check, or OK
            cp_in_region = [cp for cp in result["change_points"] if cp[1] < nok_region_limit]
            if len(cp_in_region) == 0:
                flag = "OK"
            elif len(cp_in_region) == 1:
                flag = "NOK"
                global_summary[fname].append(f"{bead_num} ({bead_type})")
            else:
                flag = "NOK_Check"
                global_summary[fname].append(f"{bead_num} ({bead_type})")

            table_data.append({
                "File": fname,
                "Bead": bead_num,
                "Bead Type": bead_type,
                "Algorithm": algorithm,
                "Change Points": len(result["change_points"]),
                "Flag": flag
            })

            if bead_num == selected_bead:
                # Add change point regions to plot
                for start, end, score in result["change_points"]:
                    if algorithm == "CUSUM":
                        raw_fig.add_vline(x=start, line=dict(color="red", width=2), opacity=0.7)
                    else:
                        raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, layer="below", line_width=0)

                raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)", line=dict(width=1)))
                color = 'red' if result["change_points"] else 'black'
                raw_fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} (filtered)", line=dict(color=color)))

                # Score visualization
                if algorithm == "Sliding Window (Original)":
                    y_scores = result["abs_scores"] if mode == "Absolute" else [v*100 for v in result["rel_scores"]]
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Score"))
                    thresh_line = threshold*100 if mode=="Relative (%)" else threshold
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=[thresh_line]*len(result["positions"]),
                                                   mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))
                    score_fig.update_layout(title="Score Trace (Change Magnitude per Window)")
                else:  # CUSUM
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=result["cusum_pos"], mode='lines', name=f"{fname} CUSUM+", line=dict(color="red")))
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=result["cusum_neg"], mode='lines', name=f"{fname} CUSUM-", line=dict(color="blue")))
                    score_fig.add_trace(go.Scatter(x=result["positions"], y=[cusum_threshold]*len(result["positions"]),
                                                   mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))
                    score_fig.update_layout(title="CUSUM Trace (Cumulative Sum)")

    raw_fig.update_layout(title=f"Raw and Filtered Signal - {algorithm}")
    st.plotly_chart(raw_fig, use_container_width=True)
    st.subheader(f"Detection Score Trace - {algorithm}")
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Change Point Summary Table")
    st.dataframe(pd.DataFrame(table_data))

    st.subheader("Global NOK and NOK_Check Beads Summary")
    if global_summary:
        global_table = pd.DataFrame([{ "File": k, "NOK/NOK_Check Beads": ", ".join(v) } for k, v in global_summary.items()])
        st.dataframe(global_table)
    else:
        st.write("✅ No NOK or NOK_Check beads detected!")
