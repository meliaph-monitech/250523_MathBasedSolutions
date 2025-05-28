import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
from scipy.stats import trim_mean

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

# --- Sharp Dip Detection ---
def has_sharp_dip(signal, baseline, min_length, min_drop_value):
    dip = baseline - signal
    i = 0
    while i < len(dip):
        if dip[i] > min_drop_value:
            start = i
            while i < len(dip) and dip[i] > min_drop_value:
                i += 1
            end = i
            run_length = end - start
            if run_length >= min_length:
                return True
        else:
            i += 1
    return False

# --- App Layout ---
st.set_page_config(page_title="Laser Welding Inspection", layout="wide")
st.title("Laser Welding Signal Analysis")

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
                    bead_metadata.append({
                        "file": os.path.basename(file),
                        "bead_number": bead_num,
                        "start_index": start,
                        "end_index": end,
                        "length": end - start + 1
                    })
                    bead_data[bead_num].append({
                        "file": os.path.basename(file),
                        "data": df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    })

            st.session_state["bead_metadata"] = bead_metadata
            st.session_state["bead_data"] = bead_data
            st.session_state["signal_column"] = signal_column
            st.success("Bead segmentation completed.")

if "bead_metadata" in st.session_state and "bead_data" in st.session_state:
    st.markdown("### Bead Segmentation Summary")
    st.dataframe(pd.DataFrame(st.session_state["bead_metadata"]))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Configuration")
    detection_mode = st.sidebar.selectbox("Detection Method", ["Sharp Dip"])
    min_dip_length = st.sidebar.slider("Min consecutive points (dip length)", 1, 50, 5)
    min_dip_value = st.sidebar.slider("Min dip amount (absolute value)", 0.01, 5.0, 0.2, 0.01)

    selected_bead = st.sidebar.selectbox("Select Bead Number to Display", sorted(st.session_state["bead_data"].keys()))
    threshold_mode = st.sidebar.selectbox(
        "Baseline method:",
        ["Global Median", "Global Mean", "Rolling Median", "Rolling Quantile (10%)", "Trimmed Mean (10%)"]
    )
    window_size = st.sidebar.slider("Rolling Window Size (for rolling methods)", 5, 100, 25)
    exclude_outliers = st.sidebar.checkbox("Exclude extreme high values when calculating baselines", value=True)

    signal_col = st.session_state["signal_column"]
    nok_files = set()
    nok_beads_by_file = defaultdict(list)

    for bead_num, entries in st.session_state["bead_data"].items():
        all_signals = [entry["data"] for entry in entries]
        min_len = min(len(sig) for sig in all_signals)
        all_signals_trimmed = [sig[:min_len] for sig in all_signals]
        stacked_signals = np.vstack(all_signals_trimmed)

        if exclude_outliers:
            p95 = np.percentile(stacked_signals, 95, axis=0)
            stacked_signals = np.minimum(stacked_signals, p95)

        if threshold_mode == "Global Median":
            baseline = np.median(stacked_signals, axis=0)
        elif threshold_mode == "Global Mean":
            baseline = np.mean(stacked_signals, axis=0)
        elif threshold_mode == "Rolling Median":
            baseline = pd.DataFrame(stacked_signals).median(axis=0).rolling(window_size, min_periods=1, center=True).median().to_numpy()
        elif threshold_mode == "Rolling Quantile (10%)":
            baseline = pd.DataFrame(stacked_signals).quantile(0.10, axis=0).rolling(window_size, min_periods=1, center=True).mean().to_numpy()
        elif threshold_mode == "Trimmed Mean (10%)":
            baseline = trim_mean(stacked_signals, proportiontocut=0.1, axis=0)
        else:
            baseline = np.median(stacked_signals, axis=0)

        for entry in entries:
            file = entry["file"]
            signal = entry["data"][:min_len]
            is_nok = has_sharp_dip(signal, baseline, min_dip_length, min_dip_value)
            if is_nok:
                nok_files.add(file)
                nok_beads_by_file[file].append(str(bead_num))

    fig = go.Figure()
    all_signals = [entry["data"] for entry in st.session_state["bead_data"][selected_bead]]
    min_len = min(len(sig) for sig in all_signals)
    all_signals_trimmed = [sig[:min_len] for sig in all_signals]
    stacked_signals = np.vstack(all_signals_trimmed)

    if exclude_outliers:
        p95 = np.percentile(stacked_signals, 95, axis=0)
        stacked_signals = np.minimum(stacked_signals, p95)

    if threshold_mode == "Global Median":
        baseline = np.median(stacked_signals, axis=0)
    elif threshold_mode == "Global Mean":
        baseline = np.mean(stacked_signals, axis=0)
    elif threshold_mode == "Rolling Median":
        baseline = pd.DataFrame(stacked_signals).median(axis=0).rolling(window_size, min_periods=1, center=True).median().to_numpy()
    elif threshold_mode == "Rolling Quantile (10%)":
        baseline = pd.DataFrame(stacked_signals).quantile(0.10, axis=0).rolling(window_size, min_periods=1, center=True).mean().to_numpy()
    elif threshold_mode == "Trimmed Mean (10%)":
        baseline = trim_mean(stacked_signals, proportiontocut=0.1, axis=0)
    else:
        baseline = np.median(stacked_signals, axis=0)

    summary = []
    for entry in st.session_state["bead_data"][selected_bead]:
        file = entry["file"]
        signal = entry["data"][:min_len]
        is_nok = has_sharp_dip(signal, baseline, min_dip_length, min_dip_value)
        color = 'red' if is_nok else 'black'
        fig.add_trace(go.Scatter(y=signal, mode='lines', name=file, line=dict(color=color)))

        summary.append({
            "File Name": file,
            "NOK": is_nok
        })

    fig.add_trace(go.Scatter(y=baseline, mode='lines', name='Baseline', line=dict(color='green', width=1, dash='dash')))

    st.markdown(f"### Signal Plot for Bead #{selected_bead}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Final Welding Result Summary")
    all_files = {entry["file"] for entry in st.session_state["bead_metadata"]}
    final_result = pd.DataFrame({
        "File Name": list(all_files),
        "Welding Result": ["NOK" if f in nok_files else "OK" for f in all_files],
        "NOK Beads": [", ".join(nok_beads_by_file.get(f, [])) for f in all_files]
    })
    st.dataframe(final_result)
