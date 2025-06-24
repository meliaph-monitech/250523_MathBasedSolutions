import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

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

# --- Process Files ---
def process_files(files, filter_column, threshold, signal_column):
    bead_data = defaultdict(list)
    for file in files:
        df = pd.read_csv(file)
        segments = segment_beads(df, filter_column, threshold)
        for bead_num, (start, end) in enumerate(segments, start=1):
            signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
            bead_data[bead_num].append((os.path.basename(file), signal))
    return bead_data

# --- Ascend Measurement ---
def measure_ascending(signal):
    x = np.arange(len(signal))
    slope = np.polyfit(x, signal, 1)[0]
    cumulative_rise = np.sum(np.maximum(0, np.diff(signal)))
    net_rise = signal.iloc[-1] - signal.iloc[0]
    return slope, cumulative_rise, net_rise

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Signal Ascent Level Analyzer")

uploaded_zip = st.sidebar.file_uploader("Upload ZIP file with welding signals", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Signal column for analysis", columns)

    if st.sidebar.button("Analyze"):
        with open("data.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extracted_files = extract_zip("data.zip", "asc_data")
        bead_data = process_files(extracted_files, filter_column, threshold, signal_column)

        selected_bead = st.selectbox("Select Bead Number", sorted(bead_data.keys()))
        table_data = []

        fig = go.Figure()
        for fname, sig in bead_data[selected_bead]:
            slope, cum_rise, net_rise = measure_ascending(sig)
            table_data.append({
                "File": fname,
                "Slope": round(slope, 4),
                "Cumulative Rise": round(cum_rise, 2),
                "Net Rise": round(net_rise, 2)
            })
            fig.add_trace(go.Scatter(y=sig, mode='lines', name=fname))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Ascent Metrics Table")
        st.dataframe(pd.DataFrame(table_data))
