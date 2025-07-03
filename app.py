
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Marketing KPI Dashboard", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

DATA_PATH = "raj excel .xlsx"
df = load_data(DATA_PATH)

st.title("📊 Marketing KPI Dashboard")

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

st.sidebar.header("🗂️ Column Mapping")
channel_col      = st.sidebar.selectbox("Channel Used",                cat_cols)
conv_col         = st.sidebar.selectbox("Conversion Rate (%)",         num_cols)
campaign_col     = st.sidebar.selectbox("Campaign Type",               cat_cols)
roi_col          = st.sidebar.selectbox("ROI (%)",                     num_cols)
cpc_col          = st.sidebar.selectbox("Cost per Click",              num_cols)
custseg_col      = st.sidebar.selectbox("Customer Segment",            cat_cols)
target_col       = st.sidebar.selectbox("Target Audience",             cat_cols)
clicks_col       = st.sidebar.selectbox("Clicks",                      num_cols)
ctr_col          = st.sidebar.selectbox("Click-through Rate (%)",      num_cols)
location_col     = st.sidebar.selectbox("Location",                    cat_cols)
acq_cost_col     = st.sidebar.selectbox("Acquisition Cost",            num_cols)
engage_col       = st.sidebar.selectbox("Engagement Score",            num_cols)
duration_col     = st.sidebar.selectbox("Campaign Duration (days)",    num_cols)

st.sidebar.success("✅ Column choices saved – scroll to see the charts!")

def _agg(data, group_col, value_col, aggfunc="mean"):
    return (
        data[[group_col, value_col]]
        .dropna()
        .groupby(group_col, as_index=False)
        .agg({value_col: aggfunc})
    )

# Row 1
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1️⃣ Channel vs Avg Conversion Rate")
    st.altair_chart(
        alt.Chart(_agg(df, channel_col, conv_col)).mark_bar().encode(
            x=alt.X(f"{channel_col}:N", title="Channel"),
            y=alt.Y(f"{conv_col}:Q", title="Average Conversion Rate"),
            tooltip=[channel_col, conv_col]
        ).properties(height=300), use_container_width=True)

with c2:
    st.subheader("2️⃣ Box-Whisker: Channel vs Conversion Rate")
    st.altair_chart(
        alt.Chart(df).mark_boxplot().encode(
            x=alt.X(f"{channel_col}:N", title="Channel"),
            y=alt.Y(f"{conv_col}:Q", title="Conversion Rate"),
            tooltip=[channel_col, conv_col]
        ).properties(height=300), use_container_width=True)

with c3:
    st.subheader("3️⃣ Campaign Type vs Avg ROI")
    st.altair_chart(
        alt.Chart(_agg(df, campaign_col, roi_col)).mark_bar().encode(
            x=alt.X(f"{campaign_col}:N", title="Campaign Type"),
            y=alt.Y(f"{roi_col}:Q", title="Average ROI"),
            tooltip=[campaign_col, roi_col]
        ).properties(height=300), use_container_width=True)

# Row 2
c4, c5, c6 = st.columns(3)
with c4:
    st.subheader("4️⃣ Avg CPC by Customer Segment")
    st.altair_chart(
        alt.Chart(_agg(df, custseg_col, cpc_col)).mark_bar().encode(
            x=alt.X(f"{custseg_col}:N", title="Customer Segment"),
            y=alt.Y(f"{cpc_col}:Q", title="Average Cost per Click"),
            tooltip=[custseg_col, cpc_col]
        ).properties(height=300), use_container_width=True)

with c5:
    st.subheader("5️⃣ Target Audience vs Avg Clicks")
    st.altair_chart(
        alt.Chart(_agg(df, target_col, clicks_col)).mark_bar().encode(
            x=alt.X(f"{target_col}:N", title="Target Audience"),
            y=alt.Y(f"{clicks_col}:Q", title="Average Clicks"),
            tooltip=[target_col, clicks_col]
        ).properties(height=300), use_container_width=True)

with c6:
    st.subheader("6️⃣ Channel vs Avg Click-through Rate")
    st.altair_chart(
        alt.Chart(_agg(df, channel_col, ctr_col)).mark_bar().encode(
            x=alt.X(f"{channel_col}:N", title="Channel"),
            y=alt.Y(f"{ctr_col}:Q", title="Average CTR"),
            tooltip=[channel_col, ctr_col]
        ).properties(height=300), use_container_width=True)

# Row 3
c7, c8, c9 = st.columns(3)
with c7:
    st.subheader("7️⃣ Location vs Avg Acquisition Cost")
    st.altair_chart(
        alt.Chart(_agg(df, location_col, acq_cost_col)).mark_bar().encode(
            x=alt.X(f"{location_col}:N", title="Location"),
            y=alt.Y(f"{acq_cost_col}:Q", title="Average Acquisition Cost"),
            tooltip=[location_col, acq_cost_col]
        ).properties(height=300), use_container_width=True)

with c8:
    st.subheader("8️⃣ Customer Segment vs Avg Engagement Score")
    st.altair_chart(
        alt.Chart(_agg(df, custseg_col, engage_col)).mark_bar().encode(
            x=alt.X(f"{custseg_col}:N", title="Customer Segment"),
            y=alt.Y(f"{engage_col}:Q", title="Average Engagement Score"),
            tooltip=[custseg_col, engage_col]
        ).properties(height=300), use_container_width=True)

with c9:
    st.subheader("9️⃣ Campaign Type vs Avg Duration (days)")
    st.altair_chart(
        alt.Chart(_agg(df, campaign_col, duration_col)).mark_bar().encode(
            x=alt.X(f"{campaign_col}:N", title="Campaign Type"),
            y=alt.Y(f"{duration_col}:Q", title="Average Duration (days)"),
            tooltip=[campaign_col, duration_col]
        ).properties(height=300), use_container_width=True)

st.caption("💡 Use the sidebar to map dataframe columns to each chart.")
