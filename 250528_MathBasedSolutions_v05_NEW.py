import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

# --- Setup ---
st.set_page_config(page_title="Laser Welding Inspection", layout="wide")
st.title("Laser Welding Signal Analysis")

# --- Sidebar: Step 1: Upload ZIP ---
zip_file = st.sidebar.file_uploader("Upload ZIP File", type="zip")

# --- Helper: Extract ZIP ---
def extract_zip(file) -> list:
    extract_dir = "extracted_csvs"
    if os.path.exists(extract_dir):
        for f in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, f))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

# --- Main Processing ---
if zip_file:
    csv_paths = extract_zip(zip_file)
    sample_df = pd.read_csv(csv_paths[0])

    # Sidebar: Step 2 - Choose filter column by index
    st.sidebar.markdown("### Filter Options")
    column_names = list(sample_df.columns)
    col_options = {f"[{i}] {name}": name for i, name in enumerate(column_names)}
    filter_col_label = st.sidebar.selectbox("Choose filter column:", list(col_options.keys()))
    filter_col = col_options[filter_col_label]

    # Sidebar: Step 3 - Input filter value
    filter_val = st.sidebar.text_input("Enter filter value (exact match):")

    # Sidebar: Step 4 - Choose signal column
    signal_col_label = st.sidebar.selectbox("Choose signal column:", list(col_options.keys()))
    signal_col = col_options[signal_col_label]

    # Sidebar: Step 5 - Button to apply filter
    if st.sidebar.button("Run Main Filter"):
        # Step 6: Process each CSV file
        bead_info = []  # to store filename, bead_number, start_index, end_index, length

        def segment_beads(df, column, threshold=0.05):
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

        file_bead_data = {}  # {file_name: list of (bead_num, start, end, df_segment)}

        for path in csv_paths:
            df = pd.read_csv(path)
            if filter_val and str(filter_val) not in df[filter_col].astype(str).unique():
                continue
            bead_segments = segment_beads(df, signal_col)
            for bead_num, (start, end) in enumerate(bead_segments, start=1):
                length = end - start + 1
                bead_info.append({
                    "File Name": os.path.basename(path),
                    "Bead Number": bead_num,
                    "Start Index": start,
                    "End Index": end,
                    "Length": length
                })
                file_bead_data.setdefault(bead_num, []).append({
                    "file": os.path.basename(path),
                    "data": df.iloc[start:end+1][signal_col].reset_index(drop=True)
                })

        # Step 6: Show expander with bead info
        with st.expander("Bead Segmentation Summary"):
            st.dataframe(pd.DataFrame(bead_info))

        # Step 7: Add threshold adjustable settings
        st.sidebar.markdown("### Threshold Settings")
        sensitivity = st.sidebar.slider("Threshold sensitivity (% below baseline)", 0, 100, 10)
        nok_percentage = st.sidebar.slider("Min % of dropped points to flag as NOK", 0, 100, 15)

        # Step 8: Choose bead number to inspect
        bead_choices = sorted(file_bead_data.keys())
        chosen_bead = st.sidebar.selectbox("Select bead number to plot:", bead_choices)

        # --- Plot and summary ---
        nok_flags = []
        fig = go.Figure()
        summary = []

        for entry in file_bead_data[chosen_bead]:
            file = entry["file"]
            signal = entry["data"]
            baseline = signal.median()
            threshold = baseline * (1 - sensitivity / 100)
            below_threshold = signal < threshold
            percent_dropped = 100 * below_threshold.sum() / len(signal)
            is_nok = percent_dropped >= nok_percentage
            color = 'red' if is_nok else 'black'

            fig.add_trace(go.Scatter(y=signal, mode='lines', name=file, line=dict(color=color)))
            summary.append({
                "File Name": file,
                "% Below Threshold": round(percent_dropped, 2),
                "NOK": is_nok
            })
            if is_nok:
                nok_flags.append(file)

        st.markdown("### NOK Summary for Selected Bead")
        st.dataframe(pd.DataFrame(summary))

        fig.update_layout(title=f"Bead #{chosen_bead} Signal Plot", xaxis_title="Time Index", yaxis_title=signal_col)
        st.plotly_chart(fig, use_container_width=True)

        # Step 9: Summary per CSV file
        full_summary = pd.DataFrame(summary)
        final_summary = full_summary.groupby("File Name")["NOK"].any().reset_index()
        final_summary["Welding Result"] = final_summary["NOK"].apply(lambda x: "NOK" if x else "OK")

        st.markdown("### Final Summary Per CSV File")
        st.dataframe(final_summary)
