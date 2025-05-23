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

    # Write the uploaded file to disk
    with open(os.path.join(extract_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract the uploaded zip file
    with zipfile.ZipFile(os.path.join(extract_dir, uploaded_file.name), 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Return the paths of the extracted CSV files
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


# --- Threshold Evaluation Optimization ---
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


# --- Normalize Signal Efficiently ---
def normalize_signal(signal, method="min-max"):
    if method == "min-max":
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) if np.max(signal) - np.min(signal) != 0 else signal
    elif method == "z-score":
        return (signal - np.mean(signal)) / np.std(signal) if np.std(signal) != 0 else signal
    else:
        return signal


# --- Main Function (Optimized for Faster Execution) ---
def process_welding_data(csv_files, thresholds, normalization_method="min-max", rule_logic="any"):
    metadata = []
    for file in csv_files:
        df = pd.read_csv(file)

        for feature_name, threshold in thresholds.items():
            # Get the index of the feature column
            feature_index = df.columns.get_loc(feature_name)
            
            segments = segment_beads(df, feature_index, threshold)
            for start, end in segments:
                signal = df.iloc[start:end+1, feature_index].values
                if normalization_method:
                    signal = normalize_signal(signal, method=normalization_method)
                features = extract_features(signal)
                status = "NOK" if evaluate_rules(features, thresholds, rule_logic) else "OK"
                metadata.append({"file": file, "bead_number": start, "status": status, "start": start, "end": end})
    
    return pd.DataFrame(metadata)


# --- Visualization of Results ---
def visualize_bead_signals(results_df, signal_column, csv_files):
    bead_numbers = sorted(results_df["bead_number"].unique())
    selected_bead = st.selectbox("Select Bead Number", bead_numbers)  # User selects the bead
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
        # Extract the zip file and get the CSV files
        csv_files = extract_zip(uploaded_file)
        st.success(f"Extracted {len(csv_files)} CSV files")

        # Initialize and configure the UI
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("LO threshold", value=0.0)
        signal_column = st.selectbox("Signal column for feature extraction", [col for col in columns if df_sample[col].dtype in [np.float64, np.int64]])

        rule_logic = st.radio("Rule logic", ["any", "all"], format_func=lambda x: "Any rule violated = NOK" if x == "any" else "All rules must be violated = NOK")

        # Define thresholds dynamically via sliders for each feature
        thresholds = {}
        for feature in ['mean', 'std', 'min', 'max', 'median', 'peak_to_peak', 'skew', 'kurtosis']:
            min_threshold = st.slider(f"{feature} - Min", -10.0, 10.0, -1.0)
            max_threshold = st.slider(f"{feature} - Max", -10.0, 10.0, 1.0)
            thresholds[feature] = (min_threshold, max_threshold)

        if st.button("Run Analysis"):
            result_df = process_welding_data(csv_files, thresholds, normalization_method="min-max", rule_logic="any")
            st.dataframe(result_df)

            # Visualize the results (Optional, add interaction for selecting bead number if needed)
            visualize_bead_signals(result_df, signal_column, csv_files)
