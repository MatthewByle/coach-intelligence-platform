import streamlit as st
import pandas as pd
import plotly.express as px
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

if "Coach" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Coach"})

stats["Coach"] = stats["Coach"].astype(str).str.strip()
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()

for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

stats["Date"] = pd.to_datetime(stats.get("Date"), errors="coerce")

# =========================================================
# SIDEBAR SELECTION
# =========================================================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

# =========================================================
# LOOKUP COACH
# =========================================================
coach_row = coaches[coaches["Head Coach"] == selected_coach].iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]
coach_games = stats[stats["Coach"] == selected_coach].copy()

# =========================================================
# DNA MODEL (SINGLE SOURCE OF TRUTH)
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

X = StandardScaler().fit_transform(coach_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
coach_features["Cluster"] = kmeans.fit_predict(X)

dna_df = coach_features.reset_index()

# =========================================================
# OPERATOR ROLE SYSTEM
# =========================================================
def get_role(c):
    return {
        0: "Balanced System Coach",
        1: "Offensive Pressure Coach",
        2: "Defensive Structure Coach",
        3: "High-Variance Transition Coach"
    }.get(c, "Unknown")

dna_df["Role"] = dna_df["Cluster"].apply(get_role)

stats = stats.merge(
    dna_df[["Coach", "Role"]],
    on="Coach",
    how="left"
)

coach_role = dna_df[dna_df["Coach"] == selected_coach]["Role"].values[0]

# =========================================================
# OPPONENT STRENGTH (SAFE)
# =========================================================
team_strength = stats.groupby("Team")[["xGF_60", "xGA_60"]].mean()

league_xgf = team_strength["xGF_60"].mean()
league_xga = team_strength["xGA_60"].mean()

team_strength["Off_Strength"] = team_strength["xGF_60"] - league_xgf
team_strength["Def_Strength"] = team_strength["xGA_60"] - league_xga

def opp_strength(row):
    opp = row.get("Opponent", None)
    if opp in team_strength.index:
        return team_strength.loc[opp, "Def_Strength"] - team_strength.loc[opp, "Off_Strength"]
    return 0

stats["Opp_Strength"] = stats.apply(opp_strength, axis=1)

cg = coach_games.copy()

if "Opp_Strength" not in cg.columns:
    cg = cg.merge(stats[["Coach", "Date", "Opp_Strength"]], on=["Coach", "Date"], how="left")

cg["Opp_Strength"] = cg["Opp_Strength"].fillna(0)

cg["Adj_xGF"] = cg["xGF_60"] - cg["Opp_Strength"] * 0.4
cg["Adj_xGA"] = cg["xGA_60"] + cg["Opp_Strength"] * 0.4

# =========================================================
# SCORECARD (FIXED + STABLE GRADING)
# =========================================================
st.subheader("Coach Scorecard")

league = stats.copy()

def pct(series, val):
    return (series <= val).mean()

off_pct = pct(league["xGF_60"].dropna(), cg["xGF_60"].mean())
def_pct = pct(league["xGA_60"].dropna(), cg["xGA_60"].mean())
sys_pct = pct(league["xG_pct"].dropna(), cg["xG_pct"].mean())

def grade(p):
    if p >= 0.85: return "A"
    if p >= 0.70: return "B"
    if p >= 0.55: return "C"
    if p >= 0.40: return "D"
    return "F"

off_grade = grade(off_pct)
def_grade = grade(def_pct)
sys_grade = grade(sys_pct)

grade_map = {"A":5,"B":4,"C":3,"D":2,"F":1}
overall = (grade_map[off_grade] + grade_map[def_grade] + grade_map[sys_grade]) / 3

overall_grade = (
    "A" if overall >= 4.5 else
    "B" if overall >= 3.5 else
    "C" if overall >= 2.5 else
    "D" if overall >= 1.5 else
    "F"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offense", off_grade)
c2.metric("Defense", def_grade)
c3.metric("System", sys_grade)
c4.metric("Overall", overall_grade)

# =========================================================
# AI NARRATIVE (RESTORED)
# =========================================================
st.subheader("AI Coaching Narrative")

st.write(f"• Role: {coach_role}")

if cg["Adj_xGF"].mean() > stats["xGF_60"].mean():
    st.write("• Above-average offensive generation after opponent adjustment.")
else:
    st.write("• Offensive output is system-dependent rather than dominant.")

if cg["Adj_xGA"].mean() < stats["xGA_60"].mean():
    st.write("• Defensive structure holds under adjusted conditions.")
else:
    st.write("• Defensive results are opponent-sensitive.")

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

team_profile = team_stats[["xGF_60","xGA_60","xG_pct","PDO"]].mean()

cols = st.columns(4)
cols[0].metric("xGF/60", round(team_profile["xGF_60"],2))
cols[1].metric("xGA/60", round(team_profile["xGA_60"],2))
cols[2].metric("xG%", round(team_profile["xG_pct"],2))
cols[3].metric("PDO", round(team_profile["PDO"],3))

# =========================================================
# SYSTEM IMPACT
# =========================================================
st.subheader("System Impact")

split = len(team_stats) // 2
before = team_stats.iloc[:split]
after = team_stats.iloc[split:]

before_xg = before["xG_pct"].mean()
after_xg = after["xG_pct"].mean()
delta = after_xg - before_xg

c1, c2, c3 = st.columns(3)
c1.metric("Before", round(before_xg,2))
c2.metric("After", round(after_xg,2))
c3.metric("Delta", round(delta,2))

# =========================================================
# DNA MAP (FIXED + CLEAN)
# =========================================================
st.subheader("Coach DNA Map")

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Coach",
    size="xG_pct"
)

selected = dna_df[dna_df["Coach"] == selected_coach]

fig.add_trace(go.Scatter(
    x=selected["xGF_60"],
    y=selected["xGA_60"],
    mode="markers+text",
    marker=dict(size=18, color="red"),
    text=[selected_coach],
    textposition="top center"
))

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coach_features.index,
    columns=coach_features.index
)

sims = dist[selected_coach].sort_values().drop(selected_coach).head(5)

st.dataframe(pd.DataFrame({
    "Coach": sims.index,
    "Score": sims.values.round(2)
}))

# =========================================================
# REPLACEMENT CANDIDATES
# =========================================================
st.subheader("Replacement Candidates")

cluster_id = dna_df[dna_df["Coach"] == selected_coach]["Cluster"].values[0]

st.dataframe(
    dna_df[dna_df["Cluster"] == cluster_id][["Coach"]]
)
