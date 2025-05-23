import numpy as np
import pandas as pd
import zipfile
import os
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
import streamlit as st

# --- Optimized File Extraction ---
def extract_zip(uploaded_file, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with open(os.path.join(extract_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    with zipfile.ZipFile(os.path.join(extract_dir, uploaded_file.name), 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]


# --- Vectorized Bead Segmentation ---
def segment_beads(df, column_index, threshold):
    signal = df.iloc[:, column_index].to_numpy()
    mask = signal > threshold
    start_indices = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
    end_indices = np.where(np.diff(mask.astype(int)) == -1)[0]
    
    if mask[-1]:
        end_indices = np.append(end_indices, len(signal) - 1)
        
    return list(zip(start_indices, end_indices))


# --- Feature Extraction for Trembling Patterns ---
def extract_features(signal):
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'peak_to_peak': np.ptp(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
    }
    
    lag = 1
    autocorr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1] if len(signal) > lag else 0
    features['autocorrelation'] = autocorr
    
    return features


# --- Threshold Evaluation ---
def evaluate_rules(features, thresholds, logic_mode):
    violations = []
    for feature, bounds in thresholds.items():
        val = features.get(feature, None)
        if val is None:
            continue
        min_val, max_val = bounds
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            violations.append(True)
        else:
            violations.append(False)
    
    return any(violations) if logic_mode == 'any' else all(violations)


# --- Normalize Signal ---
def normalize_signal(signal, method="min-max"):
    if method == "min-max":
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) if np.max(signal) - np.min(signal) != 0 else signal
    elif method == "z-score":
        return (signal - np.mean(signal)) / np.std(signal) if np.std(signal) != 0 else signal
    else:
        return signal


# --- Process Welding Data ---
def process_welding_data(csv_files, thresholds, normalization_method="min-max", rule_logic="any"):
    metadata = []
    feature_ranges = {feature: [] for feature in thresholds}
    
    for file in csv_files:
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        signal_column_name = st.session_state.filter_column
        signal_column_index = df.columns.get_loc(signal_column_name)

        for feature_name in thresholds.keys():
            feature_values = []
            for start, end in segment_beads(df, signal_column_index, 0.0):
                signal = df.iloc[start:end+1, signal_column_index].values
                features = extract_features(signal)
                feature_values.append(features[feature_name])

            feature_ranges[feature_name] = (min(feature_values), max(feature_values))
    
    for feature_name, (min_val, max_val) in feature_ranges.items():
        min_threshold = st.slider(f"{feature_name} - Min", min_val, max_val, min_val)
        max_threshold = st.slider(f"{feature_name} - Max", min_val, max_val, max_val)
        thresholds[feature_name] = (min_threshold, max_threshold)

    for file in csv_files:
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        signal_column_name = st.session_state.filter_column
        signal_column_index = df.columns.get_loc(signal_column_name)

        for feature_name, threshold in thresholds.items():
            segments = segment_beads(df, signal_column_index, threshold[0])
            for start, end in segments:
                signal = df.iloc[start:end+1, signal_column_index].values
                if normalization_method:
                    signal = normalize_signal(signal, method=normalization_method)
                features = extract_features(signal)
                status = "NOK" if evaluate_rules(features, thresholds, rule_logic) else "OK"
                metadata.append({"file": file, "bead_number": start, "status": status, "start": start, "end": end})
    
    return pd.DataFrame(metadata)


# --- Visualization of Results ---
def visualize_bead_signals(results_df, signal_column, csv_files):
    bead_numbers = sorted(results_df["bead_number"].unique())
    selected_bead = st.selectbox("Select Bead Number", bead_numbers)
    fig = go.Figure()

    for _, row in results_df[results_df["bead_number"] == selected_bead].iterrows():
        df = pd.read_csv(row["file"])
        signal = df[signal_column].iloc[row["start"]:row["end"]+1].values
        color = "red" if row["status"] == "NOK" else "black"
        fig.add_trace(go.Scatter(y=signal, mode="lines", line=dict(color=color, width=1), name=f"{row['file']} ({row['status']})"))

    fig.update_layout(title=f"Bead #{selected_bead} Signal Comparison", xaxis_title="Index", yaxis_title=signal_column)
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Laser Welding Anomaly Detection (Math Rule-Based)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP with CSVs", type="zip")
    
    if uploaded_file:
        csv_files = extract_zip(uploaded_file)
        st.session_state.csv_files = csv_files
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("LO threshold", value=0.0)

        st.session_state.filter_column = filter_column
        st.session_state.threshold = threshold

        if st.button("Segment Beads"):
            result_df = process_welding_data(csv_files, {}, normalization_method="min-max", rule_logic="any")
            st.session_state.result_df = result_df  # Store result in session state

            visualize_bead_signals(result_df, filter_column, csv_files)

# Main display for results
if "result_df" in st.session_state:
    st.subheader("Detection Results")
    st.dataframe(st.session_state.result_df[["file", "bead_number", "status"]])

    # Optionally, show the rule logic selector here
    rule_logic = st.radio("Rule logic", ["any", "all"], format_func=lambda x: "Any rule violated = NOK" if x == "any" else "All rules must be violated")
    st.session_state.rule_logic = rule_logic
