import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
import numpy as np
from collections import defaultdict

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

# --- Feature Extraction ---
def extract_advanced_features(signal):
    if len(signal) == 0:
        return [0] * 8
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak]

def extract_frequency_features(signal, n_fft=256):
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
    else:
        signal = signal[:n_fft]
    fft_vals = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_vals)[:n_fft // 2]
    avg_power = np.mean(fft_magnitude)
    peak_power = np.max(fft_magnitude)
    dom_freq_index = np.argmax(fft_magnitude)
    return [avg_power, peak_power, dom_freq_index]

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    use_freq_features = st.checkbox("Include Frequency-Domain Features", value=False)

    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())

        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)

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
                features_by_bead = defaultdict(list)
                for entry in st.session_state["metadata"]:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    time_features = extract_advanced_features(bead_segment.iloc[:, 0].values)
                    freq_features = extract_frequency_features(bead_segment.iloc[:, 0].values) if use_freq_features else []
                    all_features = time_features + freq_features
                    bead_number = entry["bead_number"]
                    features_by_bead[bead_number].append(all_features)
                st.success("Feature extraction complete")
                st.session_state["features_by_bead"] = features_by_bead
                st.session_state["use_freq_features"] = use_freq_features

        # Feature configuration
        st.subheader("Set Thresholds for Feature Extraction")

        feature_names = [
            "Mean Value", "STD Value", "Min Value", "Max Value", "Median Value",
            "Skewness", "Kurtosis", "Peak-to-Peak"
        ]
        if st.session_state.get("use_freq_features", False):
            feature_names += ["Avg FFT Power", "Peak FFT Power", "Dominant Freq Index"]

        thresholds = {}
        active_features = {}
        selected_logic = {}

        for idx, feature_name in enumerate(feature_names):
            feature_values = []
            for feature_list in st.session_state["features_by_bead"].values():
                feature_values.extend([features[idx] for features in feature_list])
            min_val, max_val = min(feature_values), max(feature_values)

            active_features[feature_name] = st.checkbox(f"Activate {feature_name}", value=False)
            if active_features[feature_name]:
                logic_option = st.selectbox(
                    f"Logic for {feature_name}",
                    options=["Within Range", "Greater Than", "Less Than", "Outside Range"],
                    key=f"logic_{feature_name}"
                )
                selected_logic[feature_name] = logic_option

                if logic_option in ["Within Range", "Outside Range"]:
                    min_thresh = st.slider(f"{feature_name} - Min", min_val, max_val, min_val)
                    max_thresh = st.slider(f"{feature_name} - Max", min_val, max_val, max_val)
                    thresholds[feature_name] = (min_thresh, max_thresh)
                else:
                    val = st.slider(f"{feature_name} Threshold", min_val, max_val, min_val)
                    thresholds[feature_name] = val

# --- Classification ---
if "features_by_bead" in st.session_state:
    st.subheader("Feature Extraction Results")

    feature_names = [
        "Mean Value", "STD Value", "Min Value", "Max Value", "Median Value",
        "Skewness", "Kurtosis", "Peak-to-Peak"
    ]
    if st.session_state.get("use_freq_features", False):
        feature_names += ["Avg FFT Power", "Peak FFT Power", "Dominant Freq Index"]

    results = []
    for bead_number, feature_list in st.session_state["features_by_bead"].items():
        metadata_entries = [entry for entry in st.session_state["metadata"] if entry["bead_number"] == bead_number]
        for i, features in enumerate(feature_list):
            classification = "OK"
            for j, feature_name in enumerate(feature_names):
                if active_features.get(feature_name, False):
                    feature_value = features[j]
                    logic = selected_logic[feature_name]
                    threshold = thresholds[feature_name]

                    if logic == "Within Range":
                        min_val, max_val = threshold
                        if not (min_val <= feature_value <= max_val):
                            classification = "NOK"
                            break
                    elif logic == "Greater Than":
                        if not (feature_value > threshold):
                            classification = "NOK"
                            break
                    elif logic == "Less Than":
                        if not (feature_value < threshold):
                            classification = "NOK"
                            break
                    elif logic == "Outside Range":
                        min_val, max_val = threshold
                        if not (feature_value < min_val or feature_value > max_val):
                            classification = "NOK"
                            break

            file_name = metadata_entries[i]["file"] if i < len(metadata_entries) else "Unknown"
            results.append({
                "File Name": os.path.basename(file_name),
                "Bead Number": bead_number,
                "Classification": classification,
                "Features": features
            })

    results_df = pd.DataFrame(results)
    with st.expander("Show All Bead Classification Results"):
        st.dataframe(results_df)
    nok_results_df = results_df[results_df["Classification"] == "NOK"]
    st.dataframe(nok_results_df)

    # --- Visualization ---
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

            match = results_df[
                (results_df["Bead Number"] == selected_bead) &
                (results_df["File Name"] == os.path.basename(file_name))
            ]

            if not match.empty:
                classification = match["Classification"].values[0]
                color = 'red' if classification == "NOK" else 'black'
            else:
                classification = "Unknown"
                color = 'gray'

            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                name=f"File: {os.path.basename(file_name)}, Bead: {selected_bead} ({classification})",
                line=dict(color=color)
            ))

        fig.update_layout(
            title=f"Bead Number {selected_bead} Signal",
            xaxis_title="Time Index",
            yaxis_title="Signal Value",
            showlegend=True
        )
        st.plotly_chart(fig)
