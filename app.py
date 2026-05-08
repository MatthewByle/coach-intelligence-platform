import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

st.title("Coach Intelligence Dashboard")

conn = st.connection("gsheets", type=GSheetsConnection)

stats = conn.read(worksheet="RawStats")
coaches = conn.read(worksheet="CoachRegistry")

st.subheader("Coaches")
st.dataframe(coaches, use_container_width=True)

st.subheader("Team Stats")
st.dataframe(stats, use_container_width=True)