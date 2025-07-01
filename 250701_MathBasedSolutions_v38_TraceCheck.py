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

# --- Core: Change Point Detection ---
def analyze_change_points(signal, window_size, step_size, threshold):
    signal = signal.dropna().reset_index(drop=True)
    change_points = []
    rel_scores, positions, slopes, spike_ratios = [], [], [], []

    for start in range(0, len(signal) - 2 * window_size + 1, step_size):
        curr = signal[start:start + window_size]
        next_ = signal[start + window_size:start + 2 * window_size]

        v1, v2 = curr.median(), next_.median()
        diff = v2 - v1

        if diff > 0:
            rel_diff = diff / max(abs(v1), 1e-6)
            rel_scores.append(rel_diff * 100)
            positions.append(start + window_size)
            slopes.append(diff / window_size)

            spike_ratio = (next_ > signal.median()).sum() / len(next_)
            spike_ratios.append(spike_ratio * 100)

            if rel_diff > (threshold / 100):
                change_points.append((start, start + 2 * window_size - 1, rel_diff * 100))
        else:
            rel_scores.append(0)
            positions.append(start + window_size)
            slopes.append(0)
            spike_ratios.append(0)

    return {
        "positions": positions,
        "rel_scores": rel_scores,
        "slopes": slopes,
        "spike_ratios": spike_ratios,
        "change_points": change_points
    }

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Refined Change Point Detector (Valley, Slope, Spike Visualization)")

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

    st.sidebar.header("Smoothing & Detection")
    use_smooth = st.sidebar.checkbox("Apply Smoothing", value=False)
    if use_smooth:
        win_len = st.sidebar.number_input("Smoothing Window Length (odd)", 3, 199, 199, step=2)
        polyorder = st.sidebar.number_input("Polynomial Order", 1, 5, 5)

    win_size = st.sidebar.number_input("Window Size (Analysis)", 10, 1000, 350, 10)
    step_size = st.sidebar.number_input("Step Size", 1, 500, 175)
    threshold = st.sidebar.number_input("Change Magnitude Threshold (%)", 1.0, 100.0, 15.0, 0.5)

    slope_discriminator = st.sidebar.slider("Slope Discriminator Threshold", 0.0000, 0.1000, 0.0001, 0.0001)
    spike_discriminator_ratio = st.sidebar.slider("Spike Discriminator Ratio (for NOK_False)", 0.05, 0.99, 0.5, 0.05)

    bead_options = sorted(raw_beads.keys())
    selected_bead = st.selectbox("Select Bead Number to Display", bead_options)

    st.subheader(f"Raw and Smoothed Signal for Bead {selected_bead}")
    raw_fig = go.Figure()
    score_fig = go.Figure()
    slope_fig = go.Figure()
    spike_fig = go.Figure()

    table_data = []
    global_summary = defaultdict(list)

    for bead_num in bead_options:
        for fname, raw_sig in raw_beads[bead_num]:
            bead_type = "Aluminum" if len(raw_sig) <= split_length else "Copper"
            sig = raw_sig.copy()

            if use_smooth and len(sig) >= win_size:
                sig = pd.Series(savgol_filter(sig, win_len, polyorder))

            result = analyze_change_points(sig, win_size, step_size, threshold)
            nok_region_limit = int(len(raw_sig) * analysis_percent / 100)

            cp_in_region = [cp for cp in result["change_points"] if cp[1] < nok_region_limit]
            signal_median = sig.median()

            flag = "OK"

            if cp_in_region:
                false_alarm = False
                for cp_start, cp_end, _ in cp_in_region:
                    window = sig.iloc[cp_start:cp_end+1]
                    high_ratio = (window > signal_median).sum() / len(window)
                    slope = (window.median() - sig.iloc[max(cp_start-1,0)]) / max((cp_end - cp_start + 1),1)

                    if high_ratio > spike_discriminator_ratio or slope < slope_discriminator:
                        false_alarm = True

                if false_alarm:
                    flag = "NOK_False"
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
                "Change Points": len(result["change_points"]),
                "Flag": flag
            })

            if bead_num == selected_bead:
                for start, end, _ in result["change_points"]:
                    raw_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, layer="below", line_width=0)
                raw_fig.add_trace(go.Scatter(y=raw_sig, mode='lines', name=f"{fname} (raw)", line=dict(width=1)))
                color = 'red' if result["change_points"] else 'black'
                raw_fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{fname} (filtered)", line=dict(color=color)))

                y_scores = result["rel_scores"]
                score_fig.add_trace(go.Scatter(x=result["positions"], y=y_scores, mode='lines+markers', name=f"{fname} Score"))
                score_fig.add_trace(go.Scatter(x=result["positions"], y=[threshold]*len(result["positions"]),
                                               mode='lines', name="Threshold", line=dict(color="orange", dash="dash")))

                slope_fig.add_trace(go.Scatter(x=result["positions"], y=result["slopes"], mode='lines+markers', name=f"{fname} Slope"))
                slope_fig.add_trace(go.Scatter(x=result["positions"], y=[slope_discriminator*100]*len(result["positions"]),
                                               mode='lines', name="Slope Threshold", line=dict(color="green", dash="dash")))

                spike_fig.add_trace(go.Scatter(x=result["positions"], y=result["spike_ratios"], mode='lines+markers', name=f"{fname} Spike Ratio"))
                spike_fig.add_trace(go.Scatter(x=result["positions"], y=[spike_discriminator_ratio*100]*len(result["positions"]),
                                               mode='lines', name="Spike Ratio Threshold", line=dict(color="purple", dash="dash")))

    st.plotly_chart(raw_fig, use_container_width=True)
    st.subheader("Score Trace (Change Magnitude per Window)")
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Slope Trace (for Tuning Slope Discriminator)")
    st.plotly_chart(slope_fig, use_container_width=True)

    st.subheader("Spike Ratio Trace (for Tuning Spike Discriminator)")
    st.plotly_chart(spike_fig, use_container_width=True)

    st.subheader("Change Point Summary Table")
    st.dataframe(pd.DataFrame(table_data))

    st.subheader("Global NOK and NOK_Check Beads Summary")
    global_table = pd.DataFrame([{ "File": k, "NOK/NOK_Check Beads": ", ".join(v) } for k, v in global_summary.items()])
    st.dataframe(global_table)
