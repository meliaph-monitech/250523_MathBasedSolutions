import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from scipy.signal import hilbert
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

# --- Streamlit App Setup ---
st.set_page_config(page_title="Signal Behavior Viewer", layout="wide")
st.title("Signal Behavior Exploration (No Statistical Thresholds)")

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

# --- Heatmap Visualization ---
if "bead_data" in st.session_state:
    st.markdown("### RMS Energy Heatmap Viewer")
    selected_bead = st.selectbox("Select Bead Number for Heatmap", sorted(st.session_state["bead_data"].keys()))
    window_size = st.slider("RMS Window Size", 5, 200, 50, 5)

    signals = st.session_state["bead_data"][selected_bead]
    file_names = [entry[0] for entry in signals]
    trimmed_signals = [entry[1] for entry in signals]
    min_len = min(len(sig) for sig in trimmed_signals)
    aligned_signals = [sig[:min_len] for sig in trimmed_signals]

    rms_matrix = []
    for sig in aligned_signals:
        rms = np.sqrt(pd.Series(sig).rolling(window=window_size, center=True, min_periods=1).mean() ** 2).to_numpy()
        rms_matrix.append(rms)

    rms_matrix = np.array(rms_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=rms_matrix,
        x=list(range(min_len)),
        y=file_names,
        colorscale='Viridis',
        colorbar=dict(title='RMS Energy'),
    ))
    fig.update_layout(
        title=f"RMS Energy Heatmap - Bead #{selected_bead}",
        xaxis_title="Time Index",
        yaxis_title="CSV File"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Envelope Plot Viewer ---
st.markdown("### Signal Envelope Viewer")
selected_bead_env = st.selectbox("Select Bead Number for Envelope Plot", sorted(st.session_state["bead_data"].keys()), key="envelope")

signals_env = st.session_state["bead_data"][selected_bead_env]
min_len_env = min(len(sig) for _, sig in signals_env)

fig_env = go.Figure()
for fname, sig in signals_env:
    sig = sig[:min_len_env]
    envelope = np.abs(hilbert(sig))
    fig_env.add_trace(go.Scatter(y=envelope, mode='lines', name=fname, line=dict(width=1)))

fig_env.update_layout(
    title=f"Envelope Plot - Bead #{selected_bead_env}",
    xaxis_title="Time Index",
    yaxis_title="Envelope Amplitude",
    showlegend=True
)
st.plotly_chart(fig_env, use_container_width=True)
