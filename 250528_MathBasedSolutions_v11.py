import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from scipy.stats import zscore

# --- File Extraction ---
def extract_zip(uploaded_file, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()

    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the ZIP file.")
        st.stop()

    return [os.path.join(extract_dir, f) for f in csv_files]

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

# --- Baseline & Peer Comparison ---
def compute_baseline(signals):
    return np.median(signals, axis=0)

def compute_rms(signal, window_size):
    return np.sqrt(pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()**2).to_numpy()

def mean_pairwise_correlation(matrix):
    n = matrix.shape[0]
    total_corr = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.corrcoef(matrix[i], matrix[j])[0, 1]
            total_corr += corr
            count += 1
    return total_corr / count if count > 0 else 1.0

# --- Streamlit App ---
st.set_page_config(page_title="Welding Anomaly Viewer", layout="wide")
st.title("Laser Welding Behavioral V11")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])

    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())

        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        signal_column = st.selectbox("Select signal column for analysis", columns)

        if st.button("Segment Beads"):
            bead_metadata = []
            bead_data = defaultdict(list)
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    bead_data[bead_num].append((os.path.basename(file), signal))
                    bead_metadata.append({
                        "file": os.path.basename(file),
                        "bead_number": bead_num,
                        "start_index": start,
                        "end_index": end,
                        "length": end - start + 1
                    })

            st.session_state["bead_data"] = bead_data
            st.session_state["bead_metadata"] = bead_metadata
            st.session_state["signal_column"] = signal_column
            st.success("Bead segmentation completed.")

# --- Visualization & Summary ---
if "bead_data" in st.session_state:
    if "bead_metadata" in st.session_state:
        st.markdown("### Bead Segmentation Summary")
        st.dataframe(pd.DataFrame(st.session_state["bead_metadata"]))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Settings")

    selected_bead = st.sidebar.selectbox("Select Bead Number to Display", sorted(st.session_state["bead_data"].keys()))
    rms_window = st.sidebar.slider("RMS Window", 5, 200, 50, 5)
    corr_threshold = st.sidebar.slider("Min Corr to Baseline", 0.0, 1.0, 0.8, 0.01)
    peer_threshold = st.sidebar.slider("Min Peer Corr", 0.0, 1.0, 0.7, 0.01)
    zscore_threshold = st.sidebar.slider("Min Energy Z-score", -5.0, 0.0, -2.0, 0.1)

    final_nok_map = defaultdict(list)
    fig = go.Figure()

    for bead_num, signal_group in st.session_state["bead_data"].items():
        min_len = min(len(sig) for _, sig in signal_group)
        signals = [sig[:min_len] for _, sig in signal_group]
        files = [fname for fname, _ in signal_group]
        matrix = np.vstack(signals)
        baseline = compute_baseline(matrix)
        peer_corr = mean_pairwise_correlation(matrix)
        baseline_rms = compute_rms(baseline, rms_window)

        if bead_num == selected_bead:
            for i, (fname, sig) in enumerate(zip(files, signals)):
                corr = np.corrcoef(sig, baseline)[0, 1]
                sig_rms = compute_rms(sig, rms_window)
                energy_diff_z = zscore(sig_rms - baseline_rms)
                dip_mask = energy_diff_z < zscore_threshold
                is_nok = (corr < corr_threshold) and (peer_corr < peer_threshold) and np.min(energy_diff_z) < zscore_threshold

                if is_nok:
                    final_nok_map[fname].append(bead_num)

                normal_y = np.where(dip_mask, np.nan, sig)
                dip_y = np.where(dip_mask, sig, np.nan)

                fig.add_trace(go.Scatter(y=normal_y, mode='lines', name=f"{fname} (normal)", line=dict(color='black')))
                fig.add_trace(go.Scatter(y=dip_y, mode='lines', name=f"{fname} (dip)", line=dict(color='red')))

            fig.add_trace(go.Scatter(y=baseline, mode='lines', name='Baseline', line=dict(color='green', dash='dash')))
            fig.update_layout(title=f"Bead #{selected_bead} Signal Overlay", xaxis_title="Index", yaxis_title="Signal")
            st.plotly_chart(fig, use_container_width=True)

    # Final Welding Result Summary
    st.markdown("### Final Welding Result Summary")
    all_files = sorted(set(m["file"] for m in st.session_state["bead_metadata"]))
    summary_data = []
    for fname in all_files:
        nok_beads = final_nok_map.get(fname, [])
        summary_data.append({
            "File Name": fname,
            "Welding Result": "NOK" if nok_beads else "OK",
            "NOK Beads": ", ".join(map(str, nok_beads)) if nok_beads else ""
        })
    st.dataframe(pd.DataFrame(summary_data))
