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
# DATA
# =========================================================
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# =========================================================
# CLEANING
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]

if "Coach" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Coach"})

stats["Coach"] = stats["Coach"].astype(str).str.strip()
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()

# numeric safety
for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

stats["Date"] = pd.to_datetime(stats.get("Date"), errors="coerce")

# =========================================================
# SIDEBAR
# =========================================================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_sidebar_coach = st.sidebar.selectbox("Select Coach", coach_list)

if st.session_state.active_coach is None:
    st.session_state.active_coach = selected_sidebar_coach

if st.sidebar.button("Reset Focus"):
    st.session_state.active_coach = selected_sidebar_coach

active_coach = st.session_state.active_coach

# =========================================================
# COACH LOOKUP
# =========================================================
filtered = coaches[coaches["Head Coach"] == active_coach]

if filtered.empty:
    st.error("Coach not found.")
    st.stop()

coach_row = filtered.iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]

# =========================================================
# SCORE ENGINE (RESTORED + FIXED)
# =========================================================
offense_score = team_stats["xGF_60"].rank(pct=True).mean() * 100
defense_score = (1 - team_stats["xGA_60"].rank(pct=True)).mean() * 100
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
# HEADER (CLEAN IDENTITY)
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write(f"Team: **{team}**")

# =========================================================
# COACH SCORECARD (RESTORED)
# =========================================================
st.subheader("Coach Scorecard")

c1, c2, c3 = st.columns(3)
c1.metric("Offense Score", round(offense_score, 1))
c2.metric("Defense Score", round(defense_score, 1))
c3.metric("Coach Score", round(coach_score, 1))

st.success(f"Grade: {grade}")

# =========================================================
# TEAM STATS (CLEAN UI - NO DROPDOWN FEEL)
# =========================================================
st.subheader("Team Stats")

coach_profile = stats[stats["Coach"] == active_coach][
    ["xGF_60", "xGA_60", "xG_pct", "PDO"]
].mean(numeric_only=True)

col1, col2, col3, col4 = st.columns(4)

col1.metric("xGF/60", round(coach_profile["xGF_60"], 2))
col2.metric("xGA/60", round(coach_profile["xGA_60"], 2))
col3.metric("xG%", round(coach_profile["xG_pct"], 2))
col4.metric("PDO", round(coach_profile["PDO"], 3))

# =========================================================
# SYSTEM IMPACT
# =========================================================
st.subheader("System Impact")

before = team_stats.head(15)
after = team_stats.tail(15)

before_xg = before["xG_pct"].mean()
after_xg = after["xG_pct"].mean()

delta = after_xg - before_xg if pd.notna(before_xg) and pd.notna(after_xg) else None

c1, c2, c3 = st.columns(3)
c1.metric("xG% Before", round(before_xg, 3) if pd.notna(before_xg) else "N/A")
c2.metric("xG% After", round(after_xg, 3) if pd.notna(after_xg) else "N/A")
c3.metric("Impact", round(delta, 3) if delta is not None else "N/A")

# =========================================================
# TEAM TREND (FULL SEASON FIXED)
# =========================================================
st.subheader("Team xG% Trend")

if "Date" in stats.columns:
    trend = team_stats.dropna(subset=["Date"]).sort_values("Date")
    st.line_chart(trend.set_index("Date")["xG_pct"])

# =========================================================
# MODEL
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

scaler = StandardScaler()
X = scaler.fit_transform(coach_features)

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
# DNA MAP (STABLE + INTERACTIVE STYLE)
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
    name="Coaches"
))

active_point = dna_df[dna_df["Coach"] == active_coach]

if not active_point.empty:
    fig.add_trace(go.Scatter(
        x=active_point["xGF_60"],
        y=active_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=[active_coach],
        textposition="top center",
        name="Selected"
    ))

fig.update_layout(hovermode="closest", legend=dict(orientation="h"))

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

if active_coach in distance_df.index:
    sims = distance_df[active_coach].sort_values().drop(active_coach).head(5)

    sim_df = pd.DataFrame({
        "Coach": sims.index,
        "Similarity": sims.values.round(2)
    })

    st.dataframe(sim_df, use_container_width=True)

# =========================================================
# REPLACEMENT CANDIDATES
# =========================================================
st.subheader("Replacement Candidates")

replacement = coach_features.sort_values("xG_pct", ascending=False).head(5)

rep_df = replacement.reset_index()[["Coach"]]

st.dataframe(rep_df, use_container_width=True)

# =========================================================
# AI NARRATIVE (RESTORED)
# =========================================================
st.subheader("Coaching Summary")

narrative = []

if coach_profile["xGF_60"] > coach_features["xGF_60"].mean():
    narrative.append("Offensive system leans aggressive and chance-driven.")
else:
    narrative.append("Offensive structure is controlled and possession-oriented.")

if coach_profile["xGA_60"] < coach_features["xGA_60"].mean():
    narrative.append("Defensive structure suppresses opponent creation.")
else:
    narrative.append("Defensive structure allows higher event volume.")

if coach_profile["xG_pct"] > coach_features["xG_pct"].mean():
    narrative.append("Team controls expected goals above league baseline.")
else:
    narrative.append("Expected goal control is below league baseline.")

for line in narrative:
    st.write("•", line)
