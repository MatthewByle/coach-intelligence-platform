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

# =========================================================
# LOOKUP
# =========================================================
coach_row = coaches[coaches["Head Coach"] == selected_coach].iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]
coach_games = stats[stats["Coach"] == selected_coach]

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
# DNA MODEL
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

X = StandardScaler().fit_transform(coach_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
coach_features["Cluster"] = kmeans.fit_predict(X)

dna_df = coach_features.reset_index()

# highlight cluster coach belongs to
selected_cluster = dna_df[dna_df["Coach"] == selected_coach]["Cluster"].values[0]

# =========================================================
# SIMILARITY MATRIX
# =========================================================
distance_df = pd.DataFrame(
    euclidean_distances(X),
    index=coach_features.index,
    columns=coach_features.index
)

# =========================================================
# LAYOUT ORDER (IMPORTANT)
# =========================================================

# -------------------------
# COACH HEADER
# -------------------------
st.subheader(f"Coach: {selected_coach}")
st.write(f"Team: **{team}**")

# -------------------------
# SCORECARD
# -------------------------
st.subheader("Coach Scorecard")

off = cg["xGF_60"].mean()
defn = cg["xGA_60"].mean()
sys = cg["xG_pct"].mean()
pdo = cg["PDO"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("xGF/60", round(off,2))
c2.metric("xGA/60", round(defn,2))
c3.metric("xG%", round(sys,2))
c4.metric("PDO", round(pdo,3))

# -------------------------
# AI NARRATIVE
# -------------------------
st.subheader("AI Coaching Narrative")

st.write(f"• This coach operates within a cluster-based tactical profile (Cluster {selected_cluster}).")

if off > coach_features["xGF_60"].mean():
    st.write("• Above-average offensive generation relative to league coaching baseline.")
else:
    st.write("• Offensive output is structurally average or system-dependent.")

if defn < coach_features["xGA_60"].mean():
    st.write("• Defensive structure suppresses chances effectively.")
else:
    st.write("• Defensive structure is more volatile and opponent-dependent.")

# -------------------------
# TEAM STATS
# -------------------------
st.subheader("Team Stats")

team_profile = team_stats[["xGF_60","xGA_60","xG_pct","PDO"]].mean(numeric_only=True)

cols = st.columns(4)
cols[0].metric("xGF/60", round(team_profile["xGF_60"],2))
cols[1].metric("xGA/60", round(team_profile["xGA_60"],2))
cols[2].metric("xG%", round(team_profile["xG_pct"],2))
cols[3].metric("PDO", round(team_profile["PDO"],3))

# -------------------------
# SYSTEM IMPACT
# -------------------------
st.subheader("System Impact (Before vs After)")

before = team_stats[team_stats["Date"] < team_stats["Date"].median()]
after = team_stats[team_stats["Date"] >= team_stats["Date"].median()]

before_xg = before["xG_pct"].mean()
after_xg = after["xG_pct"].mean()
delta = after_xg - before_xg

c1, c2, c3 = st.columns(3)
c1.metric("Before", round(before_xg,2))
c2.metric("After", round(after_xg,2))
c3.metric("Delta", round(delta,2))

# -------------------------
# TREND
# -------------------------
st.subheader("Team xG% Trend")

st.line_chart(team_stats[["Date","xG_pct"]].dropna().set_index("Date"))

# -------------------------
# DNA MAP
# -------------------------
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
    textposition="top center",
    name="Selected Coach"
))

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SIMILAR COACHES
# -------------------------
st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:
    sims = distance_df[selected_coach].sort_values().drop(selected_coach).head(5)

    st.dataframe(pd.DataFrame({
        "Coach": sims.index,
        "Similarity Score": sims.values.round(2)
    }))

# -------------------------
# REPLACEMENTS
# -------------------------
st.subheader("Replacement Candidates")

st.dataframe(
    dna_df[dna_df["Cluster"] == selected_cluster][["Coach"]]
)
