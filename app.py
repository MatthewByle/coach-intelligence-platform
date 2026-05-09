import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="NHL Coaching", layout="wide")
st.title("NHL Coaching")

# =========================================================
# LOAD DATA
# =========================================================
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# =========================================================
# CLEAN DATA
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]
stats = stats.loc[:, stats.columns != ""]

if "Date" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Date"})

stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")

for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()
coaches["Team Name"] = coaches["Team Name"].astype(str).str.strip()
stats["Coach"] = stats["Coach"].astype(str).str.strip()
stats["Team"] = stats["Team"].astype(str).str.strip()

# =========================================================
# SIDEBAR
# =========================================================
team_list = sorted(coaches["Team Name"].dropna().unique())

selected_team = st.sidebar.selectbox(
    "Select Team",
    ["All Teams"] + team_list
)

if selected_team == "All Teams":
    coach_list = sorted(coaches["Head Coach"].dropna().unique())
else:
    coach_list = sorted(
        coaches[coaches["Team Name"] == selected_team]["Head Coach"].dropna().unique()
    )

selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

# =========================================================
# COACH LOOKUP
# =========================================================
coach_row = coaches[coaches["Head Coach"] == selected_coach]

if coach_row.empty:
    st.error("Coach not found")
    st.stop()

coach_row = coach_row.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(coach_row.get("Hire Date", None), errors="coerce")

# =========================================================
# TEAM SPLIT
# =========================================================
team_data = stats[stats["Team"] == team].sort_values("Date")

before = team_data[team_data["Date"] < hire_date].tail(25)
after = team_data[team_data["Date"] >= hire_date].head(25)

before_xg = before["xG_pct"].mean() if not before.empty else None
after_xg = after["xG_pct"].mean() if not after.empty else None

delta = (
    (after_xg - before_xg)
    if before_xg is not None and after_xg is not None
    else None
)

# =========================================================
# COACH FEATURES MODEL
# =========================================================
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

# =========================================================
# 1. COACH HEADER
# =========================================================
st.subheader("Coach")
st.markdown(f"## {selected_coach}")
st.write(f"Team: **{team}**")

st.divider()

# =========================================================
# 2. COACH SCORECARD
# =========================================================
st.subheader("Coach Scorecard")

coach_games = stats[stats["Coach"] == selected_coach]

offense = coach_games["xGF_60"].mean()
defense = coach_games["xGA_60"].mean()

col1, col2 = st.columns(2)
col1.metric("Offense", round(offense, 2) if pd.notna(offense) else "N/A")
col2.metric("Defense", round(defense, 2) if pd.notna(defense) else "N/A")

st.divider()

# =========================================================
# 3. TEAM STATS
# =========================================================
st.subheader("Team Stats")

team_summary = stats[stats["Team"] == team]

col1, col2, col3 = st.columns(3)
col1.metric("xGF/60", round(team_summary["xGF_60"].mean(), 2))
col2.metric("xGA/60", round(team_summary["xGA_60"].mean(), 2))
col3.metric("PDO", round(team_summary["PDO"].mean(), 3))

st.divider()

# =========================================================
# 4. SYSTEM IMPACT
# =========================================================
st.subheader("System Impact")

col1, col2, col3 = st.columns(3)

col1.metric("xG% Before", round(before_xg, 3) if before_xg else "N/A")
col2.metric("xG% After", round(after_xg, 3) if after_xg else "N/A")
col3.metric("Impact Delta", round(delta, 3) if delta is not None else "N/A")

st.divider()

# =========================================================
# 5. TREND
# =========================================================
st.subheader("Team xG% Trend")

if not team_data.empty:
    st.line_chart(team_data.set_index("Date")["xG_pct"])

st.divider()

# =========================================================
# 6. DNA MAP (FULL HIGHLIGHT SYSTEM)
# =========================================================
st.subheader("Coach DNA Map")

dna_df = coach_features.reset_index()

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Coach"
)

# -------------------------
# SELECTED COACH (YOU)
# -------------------------
selected_point = dna_df[dna_df["Coach"] == selected_coach]

if not selected_point.empty:
    fig.add_scatter(
        x=selected_point["xGF_60"],
        y=selected_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=18),
        text=["YOU"],
        textposition="top center",
        name="Selected Coach"
    )

# -------------------------
# SIMILAR COACHES (GREEN)
# -------------------------
if selected_coach in distance_df.index:
    similar = (
        distance_df[selected_coach]
        .sort_values()
        .drop(selected_coach)
        .head(5)
        .index
    )

    sim_df = dna_df[dna_df["Coach"].isin(similar)]

    fig.add_scatter(
        x=sim_df["xGF_60"],
        y=sim_df["xGA_60"],
        mode="markers",
        marker=dict(size=14),
        name="Similar Coaches"
    )

# -------------------------
# REPLACEMENT COACHES (BLUE)
# -------------------------
replacement = coach_features.sort_values("xG_pct", ascending=False).head(5).index

rep_df = dna_df[dna_df["Coach"].isin(replacement)]

fig.add_scatter(
    x=rep_df["xGF_60"],
    y=rep_df["xGA_60"],
    mode="markers",
    marker=dict(size=14),
    name="Replacement Candidates"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 7. SIMILAR COACHES TABLE
# =========================================================
st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:
    sim = distance_df[selected_coach].sort_values().drop(selected_coach).head(5)

    st.dataframe(
        pd.DataFrame({"Coach": sim.index}),
        hide_index=True
    )

# =========================================================
# 8. REPLACEMENTS TABLE
# =========================================================
st.subheader("Replacement Candidates")

rep_table = coach_features.sort_values("xG_pct", ascending=False).head(5)

st.dataframe(
    pd.DataFrame({"Coach": rep_table.index}),
    hide_index=True
)
