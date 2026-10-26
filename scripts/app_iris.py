"""Streamlit app to explore the iris dataset.

Run with: uv run streamlit run scripts/app_iris.py
"""

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the iris dataset shipped with scikit-learn, with species names."""
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(data.target_names)))
    return df.drop(columns="target")


st.title("Iris Project Course 03 - lingxuan.zhu@edu.em-lyon.com")

df = load_data()
features = [c for c in df.columns if c != "species"]

# Sidebar filters
st.sidebar.header("Filters")
species = st.sidebar.multiselect(
    "Species", sorted(df["species"].unique()), default=sorted(df["species"].unique())
)
feature = st.sidebar.selectbox("Feature to explore", features, index=2)
df = df[df["species"].isin(species)]

# Key figures
col1, col2, col3 = st.columns(3)
col1.metric("Flowers", len(df))
col2.metric(f"Average {feature}", f"{df[feature].mean():.2f}" if len(df) else "n/a")
col3.metric(f"Max {feature}", f"{df[feature].max():.2f}" if len(df) else "n/a")

st.subheader("Data")
st.dataframe(df)

st.subheader("Average of each feature by species")
st.bar_chart(df.groupby("species")[features].mean())

st.subheader(f"{feature} vs the other features")
other = st.selectbox("Second feature", [f for f in features if f != feature])
st.scatter_chart(df, x=feature, y=other, color="species")

st.subheader(f"Distribution of {feature}")
st.bar_chart(df[feature].round(1).value_counts().sort_index())
