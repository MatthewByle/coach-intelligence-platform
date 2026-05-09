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

st.write("Coach Columns:")
st.write(coaches.columns)

# --- COACH CONTEXT ---
coach_row = coaches[coaches["Coach"] == selected_coach].iloc[0]

team = coach_row["Team"]
hire_date = coach_row["Hire_Date"]
fire_date = coach_row["Fire_Date"]

st.subheader("Coach Context")
st.write("Team:", team)
st.write("Hire Date:", hire_date)
st.write("Fire Date:", fire_date)

coaches = load_data("Coach_Registry")

st.write(coaches.columns)
