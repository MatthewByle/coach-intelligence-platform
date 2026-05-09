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
# STATE
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
selected_sidebar = st.sidebar.selectbox("Select Coach", coach_list)

if st.session_state.active_coach is None:
    st.session_state.active_coach = selected_sidebar

if st.sidebar.button("Reset Focus"):
    st.session_state.active_coach = selected_sidebar

active_coach = st.session_state.active_coach

# =========================================================
# COACH LOOKUP
# =========================================================
coach_row = coaches[coaches["Head Coach"] == active_coach].iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]
coach_games = stats[stats["Coach"] == active_coach]

# =========================================================
# 🧠 OPPONENT ADJUSTED COACHING INDEX (OACI)
# =========================================================

# estimate opponent strength from league context
team_strength = stats.groupby("Team")["xGA_60"].mean()
league_mean = team_strength.mean()
league_std = team_strength.std()

def get_opponent_strength(row):
    opp_team = row.get("Opponent", None)
    if opp_team in team_strength:
        return (team_strength[opp_team] - league_mean) / (league_std + 1e-6)
    return 0

if "Opponent" in stats.columns:
    coach_games = coach_games.copy()
    coach_games["Opp_Strength"] = coach_games.apply(get_opponent_strength, axis=1)
else:
    coach_games["Opp_Strength"] = 0

# adjusted metrics
coach_games["Adj_xGF"] = coach_games["xGF_60"] - coach_games["Opp_Strength"] * 0.5
coach_games["Adj_xGA"] = coach_games["xGA_60"] + coach_games["Opp_Strength"] * 0.5

oaci_offense = coach_games["Adj_xGF"].mean()
oaci_defense = coach_games["Adj_xGA"].mean()

# normalize into 0-100 scale
oaci_offense_score = 50 + (oaci_offense - stats["xGF_60"].mean()) / (stats["xGF_60"].std() + 1e-6) * 15
oaci_defense_score = 50 + (stats["xGA_60"].mean() - oaci_defense) / (stats["xGA_60"].std() + 1e-6) * 15

oaci_offense_score = max(0, min(100, oaci_offense_score))
oaci_defense_score = max(0, min(100, oaci_defense_score))

# =========================================================
# FINAL SCORE + GRADING
# =========================================================
offense_score = oaci_offense_score
defense_score = oaci_defense_score

coach_score = (0.6 * offense_score) + (0.4 * defense_score)

if coach_score >= 80:
    grade = "A"
elif coach_score >= 70:
    grade = "B"
elif coach_score >= 60:
    grade = "C"
elif coach_score >= 50:
    grade = "D"
else:
    grade = "F"

# =========================================================
# HEADER
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write(f"Team: **{team}**")

# =========================================================
# SCORECARD (NOW REAL)
# =========================================================
st.subheader("Coach Scorecard")

c1, c2, c3 = st.columns(3)
c1.metric("Offense (Adj)", round(offense_score, 1))
c2.metric("Defense (Adj)", round(defense_score, 1))
c3.metric("Coach Score", round(coach_score, 1))

st.success(f"Grade: {grade}")

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

profile = team_stats[["xGF_60", "xGA_60", "xG_pct", "PDO"]].mean(numeric_only=True)

cols = st.columns(4)
cols[0].metric("xGF/60", round(profile["xGF_60"], 2))
cols[1].metric("xGA/60", round(profile["xGA_60"], 2))
cols[2].metric("xG%", round(profile["xG_pct"], 2))
cols[3].metric("PDO", round(profile["PDO"], 3))

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

distance_df = pd.DataFrame(
    euclidean_distances(X),
    index=coach_features.index,
    columns=coach_features.index
)

dna_df = coach_features.reset_index()

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
    marker=dict(size=10, opacity=0.4),
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

fig.update_layout(hovermode="closest", showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

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

replacements = coach_features.sort_values("xG_pct", ascending=False).head(5)

st.dataframe(replacements.reset_index()[["Coach"]])

# =========================================================
# AI SUMMARY
# =========================================================
st.subheader("Coaching Summary")

st.write(f"""
- Offensive profile is {'above' if offense_score > 50 else 'below'} adjusted baseline  
- Defensive profile is {'strong' if defense_score > 50 else 'weak'} relative to opponent strength  
- System grade: **{grade}**
""")
