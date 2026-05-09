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
# SESSION STATE
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
# CLEAN DATA
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]

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
# =========================================================
# 🧠 OPPONENT STRENGTH MODEL (FULL VERSION)
# =========================================================
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

stats["Opp_Strength"] = stats.apply(get_opponent_strength, axis=1)

# =========================================================
# ADJUSTED COACH METRICS
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
# ROLE-BASED GRADING ENGINE
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
overall_score = (grade_map[off_grade] + grade_map[def_grade] + grade_map[sys_grade]) / 3

if overall_score >= 4.5:
    overall_grade = "A"
elif overall_score >= 3.5:
    overall_grade = "B"
elif overall_score >= 2.5:
    overall_grade = "C"
elif overall_score >= 1.5:
    overall_grade = "D"
else:
    overall_grade = "F"

# =========================================================
# HEADER
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write(f"Team: **{team}**")
st.write(f"Role: **{coach_role}**")

# =========================================================
# SCORECARD
# =========================================================
st.subheader("Coach Scorecard (Role + Opponent Adjusted)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offense", off_grade)
c2.metric("Defense", def_grade)
c3.metric("System", sys_grade)
c4.metric("Overall", overall_grade)

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

profile = team_stats[["xGF_60","xGA_60","xG_pct","PDO"]].mean(numeric_only=True)

cols = st.columns(4)
cols[0].metric("xGF/60", round(profile["xGF_60"],2))
cols[1].metric("xGA/60", round(profile["xGA_60"],2))
cols[2].metric("xG%", round(profile["xG_pct"],2))
cols[3].metric("PDO", round(profile["PDO"],3))

# =========================================================
# DNA MAP
# =========================================================
st.subheader("Coach DNA Map")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dna_df["xGF_60"],
    y=dna_df["xGA_60"],
    mode="markers",
    text=dna_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    marker=dict(size=10, opacity=0.4)
))

active_point = dna_df[dna_df["Coach"] == active_coach]

if not active_point.empty:
    fig.add_trace(go.Scatter(
        x=active_point["xGF_60"],
        y=active_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=[active_coach],
        textposition="top center"
    ))

fig.update_layout(showlegend=False, hovermode="closest")

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

distance_df = pd.DataFrame(
    euclidean_distances(X),
    index=coach_features.index,
    columns=coach_features.index
)

if active_coach in distance_df.index:
    sims = distance_df[active_coach].sort_values().drop(active_coach).head(5)

    st.dataframe(pd.DataFrame({
        "Coach": sims.index,
        "Score": sims.values.round(2)
    }))

# =========================================================
# REPLACEMENTS
# =========================================================
st.subheader("Replacement Candidates")

st.dataframe(dna_df[dna_df["Role"] == coach_role][["Coach"]])

# =========================================================
# 🧠 AI NARRATIVE (FULL SYSTEM)
# =========================================================
st.subheader("AI Coaching Narrative")

narrative = []

narrative.append(f"""
This coach operates as a **{coach_role}**, meaning their identity is structurally consistent within a defined tactical archetype.
""")

if cg["Adj_xGF"].mean() > stats["xGF_60"].mean():
    narrative.append("They generate above-average offensive output even after opponent adjustment.")
else:
    narrative.append("Offensive output remains system-dependent rather than dominance-driven.")

if cg["Adj_xGA"].mean() < stats["xGA_60"].mean():
    narrative.append("Defensive structure holds under opponent-adjusted conditions.")
else:
    narrative.append("Defensive outcomes are more opponent-sensitive than structural.")

if cg["Adj_xG_pct"].mean() > stats["xG_pct"].mean():
    narrative.append("They maintain above-average control of expected goal environment.")
else:
    narrative.append("Expected goal control fluctuates depending on matchup strength.")

for line in narrative:
    st.write("•", line)
