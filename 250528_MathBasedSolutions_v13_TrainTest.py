import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.stats import trim_mean

# --- File Extraction ---
def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
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

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Dip Valley Detector Using OK Reference")

st.sidebar.header("Upload OK Reference Data")
ok_zip = st.sidebar.file_uploader("ZIP file with ONLY OK welds", type="zip", key="ok")

st.sidebar.header("Upload Test Data")
test_zip = st.sidebar.file_uploader("ZIP file to Test", type="zip", key="test")

filter_column = st.sidebar.text_input("Column name for segmentation")
threshold = st.sidebar.number_input("Threshold for filtering", value=0.0)
signal_column = st.sidebar.text_input("Signal column for analysis")
drop_margin = st.sidebar.slider("Drop Margin (% below baseline)", 1.0, 50.0, 10.0, 0.5)
min_duration = st.sidebar.slider("Min Dip Duration (points)", 5, 500, 50, 5)
window_size = st.sidebar.slider("Rolling Window Size", 5, 200, 50, 5)

if ok_zip and test_zip and filter_column and signal_column:
    with open("ok.zip", "wb") as f:
        f.write(ok_zip.getbuffer())
    with open("test.zip", "wb") as f:
        f.write(test_zip.getbuffer())

    ok_files = extract_zip("ok.zip", "ok_data")
    test_files = extract_zip("test.zip", "test_data")

    def process_csvs(csv_files):
        bead_data = defaultdict(list)
        for file in csv_files:
            df = pd.read_csv(file)
            if filter_column not in df.columns or signal_column not in df.columns:
                continue
            segments = segment_beads(df, filter_column, threshold)
            for bead_num, (start, end) in enumerate(segments, start=1):
                sig = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                bead_data[bead_num].append((os.path.basename(file), sig))
        return bead_data

    ok_beads = process_csvs(ok_files)
    test_beads = process_csvs(test_files)

    results = []

    for bead_num in sorted(test_beads.keys()):
        if bead_num not in ok_beads:
            continue

        ok_signals = [sig[:min(len(sig) for _, sig in ok_beads[bead_num])] for _, sig in ok_beads[bead_num]]
        ok_maxes = [np.max(sig) for sig in ok_signals]
        threshold_max = np.percentile(ok_maxes, 95)

        # Remove high mountains
        clean_signals = [sig for sig, maxval in zip(ok_signals, ok_maxes) if maxval <= threshold_max]
        ok_matrix = np.vstack(clean_signals)

        # Create baseline from trimmed mean
        baseline = np.apply_along_axis(lambda x: trim_mean(x, 0.1), axis=0, arr=ok_matrix)

        for fname, sig in test_beads[bead_num]:
            sig = sig[:len(baseline)]
            dip_mask = (sig < baseline * (1 - drop_margin / 100)).astype(int)

            dip_count = 0
            current = 0
            for val in dip_mask:
                if val:
                    current += 1
                    if current >= min_duration:
                        dip_count += 1
                        break
                else:
                    current = 0

            result = {
                "File": fname,
                "Bead": bead_num,
                "Dip Detected": dip_count > 0
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    st.markdown("### Dip Detection Results")
    st.dataframe(results_df)

    summary = results_df.groupby("File")["Dip Detected"].any().reset_index()
    summary["Welding Result"] = summary["Dip Detected"].apply(lambda x: "NOK" if x else "OK")
    st.markdown("### Final File Classification")
    st.dataframe(summary)
