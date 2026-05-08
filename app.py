import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.title("Coach Dashboard")

conn = st.connection("gsheets", type=GSheetsConnection)

SPREADSHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

stats = conn.read(
    spreadsheet=SPREADSHEET_ID,
    worksheet="RawStats"
)

coaches = conn.read(
    spreadsheet=SPREADSHEET_ID,
    worksheet="Coach_Registry"
)

st.dataframe(coaches)
st.dataframe(stats)