import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="NHL Coaching", layout="wide")

st.title("NHL Coaching")

# -----------------------------
# SHEET LOADER
# -----------------------------
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq"
        f"?tqx=out:csv&sheet={sheet_name}"
    )
    return pd.read_csv(url)

# -----------------------------
# LOAD DATA
# -----------------------------
stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# -----------------------------
# CLEAN COLUMN ISSUES (CRITICAL FIX)
# -----------------------------
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

# drop junk columns like "", Unnamed: 0
stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]
stats = stats.loc[:, stats.columns != ""]

# FIX MISALIGNED DATE COLUMN (KEY FIX)
if "Date" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Date"})

# -----------------------------
# TYPE CLEANING
# -----------------------------
stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")

for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()
coaches["Team Name"] = coaches["Team Name"].astype(str).str.strip()
stats["Coach"] = stats["Coach"].astype(str).str.strip()
stats["Team"] = stats["Team"].astype(str).str.strip()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

team_options = sorted(coaches["Team Name"].dropna().unique())

selected_team = st.sidebar.selectbox(
    "Select Team",
    ["All Teams"] + team_options
)

if selected_team == "All Teams":
    coach_options = sorted(coaches["Head Coach"].dropna().unique())
else:
    coach_options = sorted(
        coaches[coaches["Team Name"] == selected_team]["Head Coach"].dropna().unique()
    )

selected_coach = st.sidebar.selectbox("Select Coach", coach_options)

# -----------------------------
# COACH LOOKUP
# -----------------------------
coach_row = coaches[coaches["Head Coach"] == selected_coach]

if coach_row.empty:
    st.error("Coach not found")
    st.stop()

coach_row = coach_row.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(coach_row.get("Hire Date", None), errors="coerce")

# -----------------------------
# TEAM DATA
# -----------------------------
team_data = stats[stats["Team"] == team].copy()
team_data = team_data.sort_values("Date")

before = team_data[team_data["Date"] < hire_date].tail(25)
after = team_data[team_data["Date"] >= hire_date].head(25)

before_xg = before["xG_pct"].mean() if not before.empty else None
after_xg = after["xG_pct"].mean() if not after.empty else None

delta = (
    (after_xg - before_xg)
    if before_xg is not None and after_xg is not None
    else None
)

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([1, 2])

# -----------------------------
# COACH PANEL
# -----------------------------
with left:
    st.subheader("Coach")
    st.markdown(f"## {selected_coach}")
    st.write(f"**Team:** {team}")

    coach_games = stats[stats["Coach"] == selected_coach]

    offense = coach_games["xGF_60"].mean()
    defense = coach_games["xGA_60"].mean()

    st.subheader("Coach Scorecard")
    st.metric("Offense", round(offense, 2) if pd.notna(offense) else "N/A")
    st.metric("Defense", round(defense, 2) if pd.notna(defense) else "N/A")

# -----------------------------
# SYSTEM IMPACT
# -----------------------------
with right:
    st.subheader("System Impact")

    col1, col2, col3 = st.columns(3)

    col1.metric("xG% Before", round(before_xg, 3) if before_xg else "N/A")
    col2.metric("xG% After", round(after_xg, 3) if after_xg else "N/A")
    col3.metric("Impact Delta", round(delta, 3) if delta is not None else "N/A")

    st.divider()

    st.subheader("Team xG% Trend")

    if not team_data.empty:
        st.line_chart(team_data.set_index("Date")["xG_pct"])

# -----------------------------
# DNA MODEL
# -----------------------------
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
coach_features["Cluster"] = kmeans.fit_predict(X_scaled)

distance_df = pd.DataFrame(
    euclidean_distances(X_scaled),
    index=coach_features.index,
    columns=coach_features.index
)

# -----------------------------
# SIMILAR COACHES
# -----------------------------
st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:
    sim = distance_df[selected_coach].sort_values().drop(selected_coach).head(5)

    st.dataframe(
        pd.DataFrame({"Coach": sim.index}),
        hide_index=True
    )

# -----------------------------
# REPLACEMENTS
# -----------------------------
st.subheader("Replacement Candidates")

if selected_coach in coach_features.index:
    rep = coach_features.sort_values("xG_pct", ascending=False).head(5)

    st.dataframe(
        pd.DataFrame({"Coach": rep.index}),
        hide_index=True
    )

# -----------------------------
# DNA MAP
# -----------------------------
st.subheader("Coach DNA Map")

fig = px.scatter(
    coach_features.reset_index(),
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Coach"
)

st.plotly_chart(fig, use_container_width=True)
