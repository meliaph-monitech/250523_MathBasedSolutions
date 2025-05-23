import numpy as np
import pandas as pd
import zipfile
import os
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from collections import defaultdict

# --- Optimized File Extraction ---
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]


# --- Vectorized Bead Segmentation ---
def segment_beads(df, column, threshold):
    signal = df[column].to_numpy()
    # Use numpy to find the indices where signal is greater than threshold
    mask = signal > threshold
    start_indices = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
    end_indices = np.where(np.diff(mask.astype(int)) == -1)[0]
    
    # Check if there is an unclosed bead at the end of the signal
    if mask[-1]:
        end_indices = np.append(end_indices, len(signal) - 1)
        
    # Pair start and end indices
    return list(zip(start_indices, end_indices))


# --- Feature Extraction for Trembling Patterns ---
def extract_features(signal):
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'peak_to_peak': np.ptp(signal),  # Peak-to-peak for fluctuations
        'skew': skew(signal),  # Asymmetry
        'kurtosis': kurtosis(signal),  # Tailedness
    }
    
    # Autocorrelation for periodicity detection
    lag = 1
    autocorr = np.corrcoef(signal[:-lag], signal[lag:])[0, 1] if len(signal) > lag else 0
    features['autocorrelation'] = autocorr
    
    return features


# --- Threshold Evaluation Optimization ---
def evaluate_rules(features, rules, logic_mode):
    violations = []
    for feature, bounds in rules.items():
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
        for feature, threshold in thresholds.items():
            segments = segment_beads(df, feature, threshold)
            for start, end in segments:
                signal = df[feature].iloc[start:end+1].values
                # Normalize signal based on the selected method
                if normalization_method:
                    signal = normalize_signal(signal, method=normalization_method)
                features = extract_features(signal)
                status = "NOK" if evaluate_rules(features, thresholds, rule_logic) else "OK"
                metadata.append({"file": file, "bead_number": start, "status": status, "start": start, "end": end})
    
    return pd.DataFrame(metadata)


# --- Visualization of Results ---
def visualize_bead_signals(results_df, signal_column, thresholds, csv_files):
    bead_numbers = sorted(results_df["bead_number"].unique())
    selected_bead = bead_numbers[0]  # You can change this to let users select a bead number
    fig = go.Figure()

    # Iterate through results and plot the signals for selected bead
    for _, row in results_df[results_df["bead_number"] == selected_bead].iterrows():
        df = pd.read_csv(row["file"])
        signal = df[signal_column].iloc[row["start"]:row["end"]+1].values
        color = "red" if row["status"] == "NOK" else "black"
        fig.add_trace(go.Scatter(y=signal, mode="lines", line=dict(color=color, width=1), name=f"{row['file']} ({row['status']})"))

    fig.update_layout(title=f"Bead #{selected_bead} Signal Comparison", xaxis_title="Index", yaxis_title=signal_column)
    fig.show()


# --- Example Usage ---
# Define the thresholds for each feature (customize as necessary)
thresholds = {
    'mean': (None, None),  # No threshold specified for simplicity
    'std': (None, None),
    'min': (None, None),
    'max': (None, None),
    'median': (None, None),
    'peak_to_peak': (None, None),
}

# Process the welding data
csv_files = extract_zip("path_to_zip_file.zip")
result_df = process_welding_data(csv_files, thresholds, normalization_method="min-max", rule_logic="any")

# Show result
print(result_df)

# Visualize the bead signal for the first bead number (you can modify for user selection)
visualize_bead_signals(result_df, "your_signal_column_name", thresholds, csv_files)
