import streamlit as st
import pandas as pd

SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

st.title("Coach Intelligence Dashboard")

# --- COACH SELECTOR ---
coach_list = coaches["Coach_Name"].dropna().unique()
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

st.write("Selected Coach:", selected_coach)