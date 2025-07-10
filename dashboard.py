import streamlit as st
import pandas as pd

st.set_page_config(page_title="Visitor Dashboard", layout="wide")
st.title("📊 Visitor Analytics Dashboard")

# Load the visitor log
try:
    df = pd.read_csv(
        "output/visitor_log.csv",
        header=0,
        names=["Visitor_ID", "Timestamp", "Event"],
        on_bad_lines='skip'  # Skip corrupted or malformed lines
    )

    # Ensure the "Event" column exists
    if "Event" not in df.columns:
        st.error("❌ 'Event' column missing in visitor_log.csv. Please ensure the file has correct headers.")
        st.stop()

    st.success("✅ Visitor log loaded successfully.")

    # Show raw table
    st.subheader("📋 Full Visitor Log")
    st.dataframe(df, use_container_width=True)

    # Total Unique Visitors
    total_visitors = df[df["Event"] == "entry"]["Visitor_ID"].nunique()

    # Currently inside
    entry_counts = df[df["Event"] == "entry"]["Visitor_ID"].value_counts()
    exit_counts = df[df["Event"] == "exit"]["Visitor_ID"].value_counts()
    currently_inside = (entry_counts - exit_counts).fillna(0)
    currently_inside = currently_inside[currently_inside > 0].count()

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("🧍 Total Unique Visitors", total_visitors)
    col2.metric("🚪 Currently Inside", currently_inside)

    # Line Chart
    st.subheader("📈 Visitor Flow Over Time")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.dropna(subset=["Timestamp"])  # Remove invalid timestamps
    chart_data = df.groupby([df["Timestamp"].dt.date, "Event"]).size().unstack(fill_value=0)
    st.line_chart(chart_data)

except FileNotFoundError:
    st.error("⚠️ visitor_log.csv not found. Run the face tracking script first.")
except Exception as e:
    st.error(f"❌ Error loading dashboard: {e}")
