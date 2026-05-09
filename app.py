import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NHL Coaching",
    layout="wide"
)

st.title("NHL Coaching")

# -----------------------------
# GOOGLE SHEET LOADER
# -----------------------------
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )
    df = pd.read_csv(url)
    return df

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# -----------------------------
# REMOVE BAD BLANK COLUMN (IMPORTANT FIX)
# -----------------------------
if stats.columns[0] == "" or "unnamed" in stats.columns[0].lower():
    stats = stats.iloc[:, 1:]

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
stats.columns = stats.columns.str.strip()
coaches.columns = coaches.columns.str.strip()

# -----------------------------
# CLEAN STRINGS
# -----------------------------
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()
coaches["Team Name"] = coaches["Team Name"].astype(str).str.strip()

stats["Coach"] = stats["Coach"].astype(str).str.strip()
stats["Team"] = stats["Team"].astype(str).str.strip()

# -----------------------------
# CONVERT DATE SAFELY
# -----------------------------
stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")

# numeric safety
for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

# -----------------------------
# SIDEBAR NAVIGATION
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

selected_coach = st.sidebar.selectbox(
    "Select Coach",
    coach_options
)

# -----------------------------
# COACH LOOKUP
# -----------------------------
coach_filtered = coaches[
    coaches["Head Coach"] == selected_coach
]

if coach_filtered.empty:
    st.error("Coach not found")
    st.stop()

coach_row = coach_filtered.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(coach_row["Hire Date"], errors="coerce")

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

    st.divider()

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
        st.line_chart(
            team_data[["Date", "xG_pct"]].dropna().set_index("Date")
        )

# -----------------------------
# DNA MODEL
# -----------------------------
coach_features = stats.groupby("Coach")[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
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
    similar = distance_df[selected_coach].sort_values().drop(selected_coach).head(5)

    st.dataframe(
        pd.DataFrame({"Coach": similar.index}),
        hide_index=True
    )

# -----------------------------
# REPLACEMENT CANDIDATES
# -----------------------------
st.subheader("Replacement Candidates")

if selected_coach in coach_features.index:
    replacements = coach_features.sort_values("xG_pct", ascending=False).head(5)

    st.dataframe(
        pd.DataFrame({"Coach": replacements.index}),
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
