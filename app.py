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

st.write("Coach Registry Columns:")
st.write(coaches.columns)

st.write("Coach Registry Preview:")
st.dataframe(coaches.head())

coach_list = coaches["Head Coach"].dropna().unique()
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

st.write("Selected Coach:", selected_coach)

coach_row = coaches[coaches["Head Coach"] == selected_coach].iloc[0]

team = coach_row["Team Name"]
hire_date = coach_row["Hire Date"]
fire_date = coach_row["Fire Date"]

st.subheader("Coach Context")
st.write("Team:", team)
st.write("Hire Date:", hire_date)
st.write("Fire Date:", fire_date)

stats["Date"] = pd.to_datetime(stats["Date"])

team_data = stats[stats["Team Name"] == team].copy()
team_data = team_data.sort_values("Date")

hire_date = pd.to_datetime(hire_date)

before = team_data[team_data["Date"] < hire_date].tail(15)
after = team_data[team_data["Date"] >= hire_date].head(15)

before_xg = before["xG%"].mean()
after_xg = after["xG%"].mean()

delta = after_xg - before_xg

st.subheader("System Impact (Before vs After)")

st.metric("xG% Before", round(before_xg, 3))
st.metric("xG% After", round(after_xg, 3))
st.metric("Impact Delta", round(delta, 3))
