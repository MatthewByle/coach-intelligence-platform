import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NHL Coaching", layout="wide")
st.title("NHL Coaching")

# =========================================================
# SESSION
# =========================================================
if "active_coach" not in st.session_state:
    st.session_state.active_coach = None

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
# CLEAN
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

if "Coach" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Coach"})

stats["Coach"] = stats["Coach"].astype(str).str.strip()
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()

for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

stats["Date"] = pd.to_datetime(stats.get("Date"), errors="coerce")

# =========================================================
# SIDEBAR
# =========================================================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)
st.session_state.active_coach = selected_coach

active_coach = st.session_state.active_coach

# =========================================================
# LOOKUP
# =========================================================
coach_row = coaches[coaches["Head Coach"] == active_coach].iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]
coach_games = stats[stats["Coach"] == active_coach].copy()

# =========================================================
# 🧠 OPPONENT STRENGTH MODEL (FIXED SAFE VERSION)
# =========================================================

team_strength = stats.groupby("Team")[["xGF_60", "xGA_60"]].mean()

league_xgf = team_strength["xGF_60"].mean()
league_xga = team_strength["xGA_60"].mean()

team_strength["Off_Strength"] = team_strength["xGF_60"] - league_xgf
team_strength["Def_Strength"] = team_strength["xGA_60"] - league_xga

def get_opponent_strength(row):
    opp = row.get("Opponent", None)
    if opp in team_strength.index:
        return (
            team_strength.loc[opp, "Def_Strength"]
            - team_strength.loc[opp, "Off_Strength"]
        )
    return 0

# IMPORTANT: add to BOTH datasets safely
stats["Opp_Strength"] = stats.apply(get_opponent_strength, axis=1)

coach_games = coach_games.merge(
    stats[["Coach", "Date", "Opp_Strength"]],
    on=["Coach", "Date"],
    how="left"
)

coach_games["Opp_Strength"] = coach_games["Opp_Strength"].fillna(0)

# =========================================================
# ADJUSTED METRICS (NOW SAFE)
# =========================================================
cg = coach_games.copy()

cg["Adj_xGF"] = cg["xGF_60"] - cg["Opp_Strength"] * 0.4
cg["Adj_xGA"] = cg["xGA_60"] + cg["Opp_Strength"] * 0.4
cg["Adj_xG_pct"] = cg["xG_pct"]

# =========================================================
# DNA MODEL
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

X = StandardScaler().fit_transform(coach_features)

coach_features["Cluster"] = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
).fit_predict(X)

dna_df = coach_features.reset_index()

# =========================================================
# ROLE SYSTEM
# =========================================================
def get_role(cluster):
    return {
        0: "Balanced System Coach",
        1: "Offensive Pressure Coach",
        2: "Defensive Structure Coach",
        3: "High-Variance Transition Coach"
    }.get(cluster, "Unknown")

dna_df["Role"] = dna_df["Cluster"].apply(get_role)

coach_role_map = dna_df[["Coach", "Role"]].copy()

stats = stats.merge(coach_role_map, on="Coach", how="left")

coach_role = coach_role_map.loc[
    coach_role_map["Coach"] == active_coach, "Role"
].values[0]

# =========================================================
# ROLE GRADING
# =========================================================
def role_grade(df, coach_name, metric, higher_is_better=True):

    coach_role = coach_role_map.loc[
        coach_role_map["Coach"] == coach_name, "Role"
    ].values[0]

    role_group = df[df["Role"] == coach_role]

    values = role_group[metric].dropna()
    coach_value = df[df["Coach"] == coach_name][metric].mean()

    if len(values) == 0:
        return "C"

    if higher_is_better:
        pct = (values <= coach_value).mean()
    else:
        pct = (values >= coach_value).mean()

    if pct >= 0.85:
        return "A"
    elif pct >= 0.70:
        return "B"
    elif pct >= 0.55:
        return "C"
    elif pct >= 0.40:
        return "D"
    else:
        return "F"

off_grade = role_grade(stats, active_coach, "xGF_60", True)
def_grade = role_grade(stats, active_coach, "xGA_60", False)
sys_grade = role_grade(stats, active_coach, "xG_pct", True)

grade_map = {"A":5,"B":4,"C":3,"D":2,"F":1}
overall = (grade_map[off_grade] + grade_map[def_grade] + grade_map[sys_grade]) / 3

overall_grade = "A" if overall >= 4.5 else "B" if overall >= 3.5 else "C" if overall >= 2.5 else "D" if overall >= 1.5 else "F"

# =========================================================
# HEADER
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write(f"Team: **{team}**")
st.write(f"Role: **{coach_role}**")

# =========================================================
# SCORECARD
# =========================================================
st.subheader("Coach Scorecard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offense", off_grade)
c2.metric("Defense", def_grade)
c3.metric("System", sys_grade)
c4.metric("Overall", overall_grade)

# =========================================================
# SUMMARY (SAFE)
# =========================================================
st.subheader("AI Coaching Narrative")

st.write(f"""
This coach operates as a **{coach_role}** with opponent-adjusted performance signals included.

- Offensive profile: {cg['Adj_xGF'].mean():.2f}
- Defensive profile: {cg['Adj_xGA'].mean():.2f}
- System consistency evaluated within role cohort
""")
