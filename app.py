import streamlit as st
import pandas as pd

st.title("Coach Intelligence Dashboard")

SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

st.subheader("Coaches")
st.dataframe(coaches)

st.subheader("Team Stats")
st.dataframe(stats)