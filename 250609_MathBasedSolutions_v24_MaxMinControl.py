# --- unchanged imports ---
import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- unchanged functions ---
def extract_zip(uploaded_file, extract_dir):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

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

# --- unchanged app setup ---
st.set_page_config(layout="wide")
st.title("Dip Valley and Rise Peak Detector V24 - Min & Max Control")

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

        st.markdown("### Bead Length Heatmaps")
        col1, col2 = st.columns(2)
        with col1:
            generate_heatmap(ok_beads, "Bead Lengths in OK ZIP")
        with col2:
            generate_heatmap(test_beads, "Bead Lengths in Test ZIP")

if "ok_beads" in st.session_state and "test_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    ok_beads = st.session_state["ok_beads"]
    test_beads = st.session_state["test_beads"]

    st.sidebar.markdown("### Lower (Dip) Detection Settings")
    drop_margin = st.sidebar.number_input("Drop Margin (% below baseline)", 0.0, 100.0, 11.0, 0.5)
    min_drop_percent = st.sidebar.number_input("Min % of points to consider as drop", 0.0, 100.0, 0.1, 0.1)
    
    use_max_drop_percent = st.sidebar.checkbox("Apply Max % for Drop", value=True)
    max_drop_percent = st.sidebar.number_input("Max % of points to consider as drop", 0.0, 100.0, 3.0, 0.5) if use_max_drop_percent else float('inf')

    min_duration = st.sidebar.number_input("Minimum Duration for Drop (consecutive points)", 1, 1000, 25, 1)
    max_duration = st.sidebar.number_input("Maximum Duration for Drop (consecutive points)", 1, 1000, 1000, 1)

    st.sidebar.markdown("### Upper (Rise) Detection Settings")
    rise_margin = st.sidebar.number_input("Rise Margin (% above baseline)", 0.0, 100.0, 40.0, 0.5)

    use_min_rise_percent = st.sidebar.checkbox("Apply Min % for Rise", value=True)
    min_rise_percent = st.sidebar.number_input("Min % of points to consider as rise", 0.0, 100.0, 0.1, 0.1) if use_min_rise_percent else 0.0

    use_max_rise_percent = st.sidebar.checkbox("Apply Max % for Rise", value=True)
    max_rise_percent = st.sidebar.number_input("Max % of points to consider as rise", 0.0, 100.0, 3.0, 0.5) if use_max_rise_percent else float('inf')

    min_rise_duration = st.sidebar.number_input("Minimum Duration for Rise (consecutive points)", 1, 1000, 15, 1)
    max_rise_duration = st.sidebar.number_input("Maximum Duration for Rise (consecutive points)", 1, 1000, 20, 1)

    selected_bead = st.selectbox("Select Bead Number to Display", sorted(ok_beads.keys()))

    drop_nok_files = defaultdict(list)
    rise_nok_files = defaultdict(list)
    drop_summary = []
    rise_summary = []
    beadwise_baselines = {}

    for bead_num in sorted(test_beads.keys()):
        ok_signals = [sig[:min(len(s) for _, s in ok_beads[bead_num])] for _, sig in ok_beads.get(bead_num, []) if bead_num in ok_beads]
        if not ok_signals:
            continue
        ok_matrix = np.vstack(ok_signals)
        baseline = np.median(ok_matrix, axis=0)
        lower_line = baseline * (1 - drop_margin / 100)
        upper_line = baseline * (1 + rise_margin / 100)
        beadwise_baselines[bead_num] = (lower_line, upper_line)

        for fname, sig in test_beads[bead_num]:
            min_len = min(len(sig), len(lower_line), len(upper_line))
            sig = sig[:min_len]
            lower = lower_line[:min_len]
            upper = upper_line[:min_len]

            below = sig < lower
            above = sig > upper

            # Drop detection
            consecutive_drops = 0
            max_consecutive_drops = 0
            for i in range(1, len(below)):
                if below[i] and below[i - 1]:
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0
                max_consecutive_drops = max(max_consecutive_drops, consecutive_drops)
            percent_below = 100 * np.sum(below) / len(sig)
            drop_triggered = (
                min_drop_percent <= percent_below <= max_drop_percent and
                min_duration <= max_consecutive_drops <= max_duration
            )
            if drop_triggered:
                drop_nok_files[fname].append(bead_num)
            if bead_num == selected_bead:
                drop_summary.append({"File": fname, "Bead": bead_num, "% Below": round(percent_below, 2), "NOK": drop_triggered})

            # Rise detection
            consecutive_rises = 0
            max_consecutive_rises = 0
            for i in range(1, len(above)):
                if above[i] and above[i - 1]:
                    consecutive_rises += 1
                else:
                    consecutive_rises = 0
                max_consecutive_rises = max(max_consecutive_rises, consecutive_rises)
            percent_above = 100 * np.sum(above) / len(sig)
            rise_triggered = (
                min_rise_percent <= percent_above <= max_rise_percent and
                min_rise_duration <= max_consecutive_rises <= max_rise_duration
            )
            if rise_triggered:
                rise_nok_files[fname].append(bead_num)
            if bead_num == selected_bead:
                rise_summary.append({"File": fname, "Bead": bead_num, "% Above": round(percent_above, 2), "NOK": rise_triggered})

    # Plotting
    fig = go.Figure()
    lower_line, upper_line = beadwise_baselines.get(selected_bead)
    for fname, sig in ok_beads.get(selected_bead, []):
        sig = sig[:min(len(s) for _, s in ok_beads[selected_bead])]
        fig.add_trace(go.Scatter(y=sig, mode='lines', line=dict(color='gray', width=1), name=f"OK: {fname}"))

    for fname, sig in test_beads.get(selected_bead, []):
        min_len = min(len(sig), len(lower_line), len(upper_line))
        sig = sig[:min_len]
        triggered_drop = fname in drop_nok_files and selected_bead in drop_nok_files[fname]
        triggered_rise = fname in rise_nok_files and selected_bead in rise_nok_files[fname]
        if triggered_drop and triggered_rise:
            color = 'purple'
        elif triggered_drop:
            color = 'red'
        elif triggered_rise:
            color = 'orange'
        else:
            color = 'black'
        fig.add_trace(go.Scatter(y=sig, mode='lines', line=dict(color=color, width=1.5), name=f"Test: {fname}"))

    fig.add_trace(go.Scatter(y=lower_line[:min_len], mode='lines', name='Lower Reference', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(y=upper_line[:min_len], mode='lines', name='Upper Reference', line=dict(color='blue', dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

    # Final Summary
    all_files = sorted({fname for bead_entries in test_beads.values() for fname, _ in bead_entries})
    final_summary = []
    for fname in all_files:
        drop_beads = drop_nok_files.get(fname, [])
        rise_beads = rise_nok_files.get(fname, [])
        all_nok_beads = sorted(set(drop_beads + rise_beads))
        final_summary.append({
            "File Name": fname,
            "NOK Beads": ", ".join(map(str, all_nok_beads)),
            "Welding Result": "NOK" if all_nok_beads else "OK",
            "Upper NOK": ", ".join(map(str, sorted(set(rise_beads) - set(drop_beads)))),
            "Lower NOK": ", ".join(map(str, sorted(set(drop_beads) - set(rise_beads)))),
            "Both NOK": ", ".join(map(str, sorted(set(drop_beads) & set(rise_beads))))
        })

    st.markdown("### Final Welding Result Summary")
    st.dataframe(pd.DataFrame(final_summary))

    st.markdown("### Drop Summary Table")
    st.dataframe(pd.DataFrame(drop_summary))

    st.markdown("### Rise Summary Table")
    st.dataframe(pd.DataFrame(rise_summary))
