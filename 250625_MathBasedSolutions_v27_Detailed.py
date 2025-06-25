import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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

# --- Trend Analysis ---
def analyze_trend_windows(signal: pd.Series, window_size: int, step_size: int, metric: str, threshold: float):
    signal = signal.dropna().reset_index(drop=True)
    total_windows = 0
    ascending_windows = 0
    score_list = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        if len(window) < window_size:
            continue

        if metric == "Linear Regression Slope":
            x = np.arange(window_size).reshape(-1, 1)
            y = window.values.reshape(-1, 1)
            slope = LinearRegression().fit(x, y).coef_[0][0]
            score = slope
        elif metric == "Delta":
            score = window.iloc[-1] - window.iloc[0]
        elif metric == "Mean Gradient":
            score = np.mean(np.diff(window.values))
        else:
            raise ValueError("Invalid metric")

        score_list.append(score)
        if score > threshold:
            ascending_windows += 1
        total_windows += 1

    percent_ascending = (ascending_windows / total_windows) * 100 if total_windows > 0 else 0

    return {
        "total_windows": total_windows,
        "ascending_windows": ascending_windows,
        "percent_ascending": percent_ascending,
        "score_list": score_list
    }

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Ascending Trend Detector v1")

st.sidebar.header("Upload Data")
test_zip = st.sidebar.file_uploader("ZIP file to Analyze", type="zip")

if test_zip:
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        sample_csv_name = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file_raw:
            sample_file = pd.read_csv(sample_file_raw)
    columns = sample_file.columns.tolist()

    filter_column = st.sidebar.selectbox("Select column for segmentation", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Select signal column for trend analysis", columns)

    if st.sidebar.button("Segment Beads"):
        with open("test.zip", "wb") as f:
            f.write(test_zip.getbuffer())

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

        test_beads = process_files(test_files)
        st.session_state["test_beads"] = test_beads
        st.session_state["analysis_ready"] = True
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

        st.markdown("### Bead Length Heatmap")
        generate_heatmap(test_beads, "Bead Lengths in Test ZIP")

if "test_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    test_beads = st.session_state["test_beads"]

    st.sidebar.header("Trend Detection Settings")
    window_size = st.sidebar.number_input("Window Size (points)", min_value=10, value=100, step=10)
    step_size = st.sidebar.number_input("Step Size (points)", min_value=1, value=20, step=1)
    metric = st.sidebar.selectbox("Trend Metric", ["Linear Regression Slope", "Delta", "Mean Gradient"])
    # threshold = st.sidebar.number_input("Trend Threshold", value=0.2, step=0.1)
    # max_percent_ascending = st.sidebar.number_input("Max % Ascending Windows Allowed", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    threshold = float(st.sidebar.text_input("Trend Threshold (exact value)", value="0.20000"))
    max_percent_ascending = float(st.sidebar.text_input("Max % Ascending Windows Allowed", value="10.00000"))

    selected_bead = st.selectbox("Select Bead Number to Display", sorted(test_beads.keys()))

    final_summary = []
    fig = go.Figure()

    for fname, signal in test_beads[selected_bead]:
        result = analyze_trend_windows(signal, window_size, step_size, metric, threshold)
        nok = result["percent_ascending"] > max_percent_ascending
        color = 'red' if nok else 'black'

        fig.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            name=fname,
            line=dict(color=color, width=1.5)
        ))

        score_stats = pd.Series(result["score_list"]).describe()

        final_summary.append({
            "File Name": fname,
            "Bead Number": selected_bead,
            "Total Windows": result["total_windows"],
            "Ascending Windows": result["ascending_windows"],
            "% Ascending": round(result["percent_ascending"], 2),
            "Result": "NOK" if nok else "OK",
            f"{metric} Max": round(score_stats["max"], 4),
            f"{metric} Mean": round(score_stats["mean"], 4),
            f"{metric} Min": round(score_stats["min"], 4)
        })

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Final Trend Summary Table")
    st.dataframe(pd.DataFrame(final_summary))
