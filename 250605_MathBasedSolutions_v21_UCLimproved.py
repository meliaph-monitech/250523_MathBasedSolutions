import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Dip and Rise Valley Detector Using OK Reference")

st.sidebar.header("Upload Data")
ok_zip = st.sidebar.file_uploader("ZIP file with ONLY OK welds", type="zip")
test_zip = st.sidebar.file_uploader("ZIP file to Test", type="zip")

if ok_zip and test_zip:
    with zipfile.ZipFile(ok_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Select column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Select signal column for analysis", columns)

    if st.sidebar.button("Segment Beads"):
        with open("ok.zip", "wb") as f:
            f.write(ok_zip.getbuffer())
        with open("test.zip", "wb") as f:
            f.write(test_zip.getbuffer())

        ok_files = extract_zip("ok.zip", "ok_data")
        test_files = extract_zip("test.zip", "test_data")

        def process_files(files):
            bead_data = defaultdict(list)
            for file in files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
                    bead_data[bead_num].append((os.path.basename(file), signal))
            return bead_data

        ok_beads = process_files(ok_files)
        test_beads = process_files(test_files)
        st.session_state["ok_beads"] = ok_beads
        st.session_state["test_beads"] = test_beads
        st.success("✅ Bead segmentation completed.")

        def generate_heatmap(bead_data, title):
            bead_lengths = defaultdict(lambda: defaultdict(int))
            bead_nums = set()
            file_names = set()
            for bead_num, entries in bead_data.items():
                for fname, sig in entries:
                    bead_lengths[fname][bead_num] = len(sig)
                    bead_nums.add(bead_num)
                    file_names.add(fname)
            bead_nums = sorted(bead_nums)
            file_names = sorted(file_names)
            heatmap_data = np.zeros((len(file_names), len(bead_nums)))
            for i, fname in enumerate(file_names):
                for j, bead in enumerate(bead_nums):
                    heatmap_data[i, j] = bead_lengths[fname].get(bead, 0)
            df_hm = pd.DataFrame(heatmap_data, index=file_names, columns=bead_nums)
            fig, ax = plt.subplots(figsize=(max(6, len(bead_nums)), max(6, len(file_names)*0.4)))
            sns.heatmap(df_hm, annot=False, cmap="YlGnBu", ax=ax, cbar=True)
            ax.set_title(title)
            ax.set_xlabel("Bead Number")
            ax.set_ylabel("File Name")
            st.pyplot(fig)

        st.markdown("### Bead Length Heatmaps")
        col1, col2 = st.columns(2)
        with col1:
            generate_heatmap(ok_beads, "Bead Lengths in OK ZIP")
        with col2:
            generate_heatmap(test_beads, "Bead Lengths in Test ZIP")

if "ok_beads" in st.session_state and "test_beads" in st.session_state:
    ok_beads = st.session_state["ok_beads"]
    test_beads = st.session_state["test_beads"]

    # --- Sidebar Controls ---
    st.sidebar.markdown("### Lower (Dip) Detection Settings")
    drop_margin = st.sidebar.slider("Drop Margin (% below baseline)", 1.0, 50.0, 11.0, 0.5)
    min_drop_percent = st.sidebar.slider("Min % of points to consider as drop", 0.1, 50.0, 0.1, 0.1)
    min_duration = st.sidebar.slider("Minimum Duration for Drop (consecutive points)", 10, 200, 25, 5)

    st.sidebar.markdown("### Upper (Rise) Detection Settings")
    rise_margin = st.sidebar.slider("Rise Margin (% above baseline)", 1.0, 50.0, 40.0, 0.5)
    min_rise_percent = st.sidebar.slider("Min % of points to consider as rise", 0.1, 50.0, 0.1, 0.1)
    max_rise_percent = st.sidebar.slider("Max % of points to consider as rise", 10.0, 100.0, 10.0, 0.5)
    min_rise_duration = st.sidebar.slider("Minimum Duration for Rise (consecutive points)", 10, 200, 30, 5)
    max_rise_duration = st.sidebar.slider("Maximum Duration for Rise (consecutive points)", 10, 200, 50, 5)
