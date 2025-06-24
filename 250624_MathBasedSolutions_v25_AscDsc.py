import streamlit as st
import zipfile, os
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- Utility Functions ---
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
    start_indices, end_indices = [], []
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

def process_files(files, filter_column, signal_column, threshold):
    bead_data = defaultdict(list)
    for file in files:
        df = pd.read_csv(file)
        segments = segment_beads(df, filter_column, threshold)
        for bead_num, (start, end) in enumerate(segments, start=1):
            signal = df.iloc[start:end+1][signal_column].reset_index(drop=True)
            bead_data[bead_num].append((os.path.basename(file), signal))
    return bead_data

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Climb/Descend Stability Detector")

st.sidebar.header("Upload and Settings")
test_zip = st.sidebar.file_uploader("Upload ZIP of Test CSVs", type="zip")

if test_zip:
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        sample_csv_name = [n for n in zip_ref.namelist() if n.endswith('.csv')][0]
        with zip_ref.open(sample_csv_name) as sample_file:
            sample_df = pd.read_csv(sample_file)

    columns = sample_df.columns.tolist()
    filter_column = st.sidebar.selectbox("Column for segmentation (threshold logic)", columns)
    threshold = st.sidebar.number_input("Segmentation threshold", value=0.0)
    signal_column = st.sidebar.selectbox("Signal column for climb analysis", columns)

    # Climb/Descend detection params
    st.sidebar.markdown("### Detection Parameters")
    CLIMB_DIFF_THRESH = st.sidebar.number_input("Climb Threshold (late_mean - early_mean)", value=0.5)
    DESCEND_DIFF_THRESH = st.sidebar.number_input("Descend Threshold (early_mean - late_mean)", value=0.5)
    SLOPE_THRESH = st.sidebar.number_input("Min absolute Slope", value=0.01)
    STABLE_STD_THRESH = st.sidebar.number_input("Max STD for Stability", value=0.2)

    if st.sidebar.button("Run Climb/Descend Analysis"):
        with open("test.zip", "wb") as f:
            f.write(test_zip.getbuffer())
        test_files = extract_zip("test.zip", "test_data")

        test_beads = process_files(test_files, filter_column, signal_column, threshold)
        st.session_state["test_beads"] = test_beads
        st.session_state["analysis_ready"] = True
        st.success("✅ Bead segmentation and signal extraction completed.")

# --- Climb/Descend Analysis ---
if "test_beads" in st.session_state and st.session_state.get("analysis_ready", False):
    test_beads = st.session_state["test_beads"]
    selected_bead = st.selectbox("Select Bead Number", sorted(test_beads.keys()))

    status_map = defaultdict(str)
    climb_summary = []

    for bead_num in sorted(test_beads.keys()):
        for fname, sig in test_beads[bead_num]:
            y = sig.values
            x = np.arange(len(y)).reshape(-1, 1)

            model = LinearRegression().fit(x, y)
            slope = model.coef_[0]

            early = y[:int(0.1 * len(y))]
            late = y[-int(0.1 * len(y)):]
            early_mean = np.mean(early)
            late_mean = np.mean(late)
            late_std = np.std(late)

            is_climb = (late_mean - early_mean > CLIMB_DIFF_THRESH) and (slope > SLOPE_THRESH)
            is_descend = (early_mean - late_mean > DESCEND_DIFF_THRESH) and (slope < -SLOPE_THRESH)
            is_stable = late_std < STABLE_STD_THRESH

            trend = "Climb" if is_climb else "Descend" if is_descend else "Flat"
            result = "NOK" if (is_climb or is_descend) and not is_stable else "OK"
            status_map[(fname, bead_num)] = result

            if bead_num == selected_bead:
                climb_summary.append({
                    "File": fname,
                    "Bead": bead_num,
                    "Early Mean": round(early_mean, 3),
                    "Late Mean": round(late_mean, 3),
                    "Slope": round(slope, 4),
                    "Late STD": round(late_std, 3),
                    "Trend": trend,
                    "Result": result
                })

    # --- Line Plot ---
    fig = go.Figure()
    for fname, sig in test_beads[selected_bead]:
        color = "red" if status_map[(fname, selected_bead)] == "NOK" else "black"
        fig.add_trace(go.Scatter(y=sig, mode='lines', name=f"{status_map[(fname, selected_bead)]}: {fname}", line=dict(color=color)))
    fig.update_layout(title=f"Bead {selected_bead} Signal Plot", xaxis_title="Time", yaxis_title=signal_column)
    st.plotly_chart(fig, use_container_width=True)

    # --- Table ---
    st.markdown("### 📋 Climb/Descend Summary Table")
    st.dataframe(pd.DataFrame(climb_summary))
