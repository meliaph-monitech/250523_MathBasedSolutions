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
        
    drop_margin = None
min_drop_percent = None
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

        selected_bead = st.selectbox("Select Bead Number to Display", sorted(ok_beads.keys()))

        # Build baseline from OK data
        ok_signals = [sig[:min(len(s) for _, s in ok_beads[selected_bead])] for _, sig in ok_beads[selected_bead]]
        ok_matrix = np.vstack(ok_signals)
        baseline = np.median(ok_matrix, axis=0)
        lower_line = baseline * (1 - drop_margin / 100)

        # Plot
        fig = go.Figure()

        # OK signals (gray)
        for fname, sig in ok_beads[selected_bead]:
            sig = sig[:len(lower_line)]
            fig.add_trace(go.Scatter(y=sig, mode='lines', line=dict(color='gray', width=1), name=f"OK: {fname}"))

        # Test signals
        drop_summary = []
        nok_files = defaultdict(list)

        for fname, sig in test_beads.get(selected_bead, []):
            sig = sig[:len(lower_line)]
            below = sig < lower_line
            percent_below = 100 * np.sum(below) / len(sig)
            color = 'red' if percent_below >= min_drop_percent else 'black'
            fig.add_trace(go.Scatter(y=sig, mode='lines', line=dict(color=color, width=1.5), name=f"Test: {fname}"))
            drop_summary.append({"File": fname, "Bead": selected_bead, "% Below": round(percent_below, 2), "NOK": percent_below >= min_drop_percent})
            if percent_below >= min_drop_percent:
                nok_files[fname].append(selected_bead)

        fig.add_trace(go.Scatter(y=lower_line, mode='lines', name='Lower Reference', line=dict(color='green', dash='dash')))

        st.plotly_chart(fig, use_container_width=True)

        # --- Heatmaps of Bead Lengths ---
        st.markdown("### Bead Length Heatmaps")
        import matplotlib.pyplot as plt
        import seaborn as sns

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

        col1, col2 = st.columns(2)
        with col1:
            generate_heatmap(ok_beads, "Bead Lengths in OK ZIP")
        with col2:
            generate_heatmap(test_beads, "Bead Lengths in Test ZIP")

        drop_margin = st.sidebar.slider("Drop Margin (% below baseline)", 1.0, 50.0, 10.0, 0.5)
        min_drop_percent = st.sidebar.slider("Min % of points to consider as drop", 0.1, 50.0, 10.0, 0.1)

        st.markdown("### Drop Summary Table")
        st.dataframe(pd.DataFrame(drop_summary))

        st.markdown("### Final Welding Result Summary")
        final_summary = pd.DataFrame({
            "File Name": sorted(set(f for f, _ in test_beads[selected_bead])),
            "NOK Beads": [", ".join(map(str, nok_files[f])) if f in nok_files else "" for f in sorted(set(f for f, _ in test_beads[selected_bead]))],
            "Welding Result": ["NOK" if f in nok_files else "OK" for f in sorted(set(f for f, _ in test_beads[selected_bead]))]
        })
        st.dataframe(final_summary)
