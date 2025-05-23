import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
import numpy as np
from collections import defaultdict  # Ensure this import is present

# --- Optimized File Extraction ---
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
    """Segments data into beads based on a threshold."""
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


# --- Feature Extraction ---
def extract_advanced_features(signal):
    """Extracts statistical features from a signal."""
    n = len(signal)
    if n == 0:
        return [0] * 8  # Adjust based on features you're extracting

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak]


# --- Process Welding Data ---
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

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

        # Segmentation and feature extraction
        if st.button("Segment Beads"):
            bead_segments = {}
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                if segments:
                    bead_segments[file] = segments
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
            st.success("Bead segmentation complete")
            st.session_state["metadata"] = metadata

        if st.button("Run Feature Extraction") and "metadata" in st.session_state:
            with st.spinner("Extracting features..."):
                features_by_bead = defaultdict(list)  # Ensure this is correctly initialized
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    features = extract_advanced_features(bead_segment.iloc[:, 0].values)
                    bead_number = entry["bead_number"]
                    features_by_bead[bead_number].append(features)
                st.success("Feature extraction complete")
                st.session_state["features_by_bead"] = features_by_bead

if "features_by_bead" in st.session_state:
    st.subheader("Feature Extraction Results")
    # Display the extracted features for each bead
    results_df = pd.DataFrame([
        {"File Name": entry["file"], "Bead Number": entry["bead_number"], "Features": features}
        for entry, features in zip(st.session_state["metadata"], st.session_state["features_by_bead"].values())
    ])
    st.dataframe(results_df)

    # --- Visualize the bead signals
    bead_numbers = sorted(set(entry["bead_number"] for entry in st.session_state["metadata"]))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)

    if selected_bead:
        fig = go.Figure()
        selected_bead_data = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == selected_bead]

        for bead_info in selected_bead_data:
            file_name = bead_info["file"]
            start_idx = bead_info["start_index"]
            end_idx = bead_info["end_index"]

            df = pd.read_csv(file_name)
            signal = df.iloc[start_idx:end_idx + 1, 0].values

            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                name=f"File: {file_name}, Bead: {selected_bead}"
            ))

        fig.update_layout(
            title=f"Bead Number {selected_bead} Signal",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
