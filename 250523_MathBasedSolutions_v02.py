import numpy as np
import pandas as pd
import zipfile
import os
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

# --- Batch Feature Extraction (Simplified) ---
def extract_features(signal):
    # Vectorized calculations for basic features (mean, std, min, max, median, peak-to-peak)
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'peak_to_peak': np.ptp(signal),  # peak-to-peak is just max - min
    }
    
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

# --- Example Usage ---
# Define the thresholds for each feature
thresholds = {
    'mean': (None, None),  # Example for no threshold, adjust as needed
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
