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
# SESSION STATE (SCOUTING ENGINE)
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
# CLEAN DATA (IMPORTANT FIXES)
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]
stats = stats.loc[:, stats.columns != ""]

if "Coach" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Coach"})

stats["Coach"] = stats["Coach"].astype(str).str.strip()
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()

# =========================================================
# SIDEBAR CONTROL
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
    st.error("Coach not found in registry.")
    st.stop()

coach_row = filtered.iloc[0]

team = coach_row["Team Name"]

# =========================================================
# TEAM FILTERED STATS
# =========================================================
team_data = stats[stats["Team"] == team].copy()

for col in ["xG_pct", "xGF_60", "xGA_60", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

# =========================================================
# SCORE ENGINE (REACTIVE)
# =========================================================
team_stats = stats[stats["Team"] == team]

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
# TOP LAYOUT ORDER (CLEAN)
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write("Team:", team)

# =========================================================
# SCORECARD (DYNAMIC FIXED)
# =========================================================
st.subheader("Coach Scorecard")

col1, col2, col3 = st.columns(3)

col1.metric("Offense Score", round(offense_score, 1))
col2.metric("Defense Score", round(defense_score, 1))
col3.metric("Coach Score", round(coach_score, 1))

st.success(f"Grade: {grade}")

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

coach_profile = stats[stats["Coach"] == active_coach].mean(numeric_only=True)

st.write({
    "xGF/60": round(coach_profile.get("xGF_60", 0), 2),
    "xGA/60": round(coach_profile.get("xGA_60", 0), 2),
    "xG%": round(coach_profile.get("xG_pct", 0), 2),
    "PDO": round(coach_profile.get("PDO", 0), 3)
})

# =========================================================
# SYSTEM IMPACT (SAFE)
# =========================================================
st.subheader("System Impact")

before = team_data.head(15)
after = team_data.tail(15)

before_xg = before["xG_pct"].mean() if "xG_pct" in before else None
after_xg = after["xG_pct"].mean() if "xG_pct" in after else None

delta = (after_xg - before_xg) if pd.notna(before_xg) and pd.notna(after_xg) else None

col1, col2, col3 = st.columns(3)

col1.metric("xG% Before", round(before_xg, 3) if pd.notna(before_xg) else "N/A")
col2.metric("xG% After", round(after_xg, 3) if pd.notna(after_xg) else "N/A")
col3.metric("Impact", round(delta, 3) if delta is not None else "N/A")

# =========================================================
# TEAM TREND (FULL SEASON FIX)
# =========================================================
st.subheader("Team xG% Trend")

if "Date" in stats.columns:
    stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")
    team_data = stats[stats["Team"] == team].dropna(subset=["Date"])

    st.line_chart(team_data.set_index("Date")["xG_pct"])

# =========================================================
# MODEL (DNA)
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

coach_features["Cluster"] = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
).fit_predict(X_scaled)

distance_df = pd.DataFrame(
    euclidean_distances(X_scaled),
    index=coach_features.index,
    columns=coach_features.index
)

dna_df = coach_features.reset_index()

# =========================================================
# DNA MAP (FULL SCOUTING INTERACTIVE)
# =========================================================
st.subheader("Coach DNA Map (Scouting Mode)")

fig = go.Figure()

# BASE
fig.add_trace(go.Scatter(
    x=dna_df["xGF_60"],
    y=dna_df["xGA_60"],
    mode="markers",
    text=dna_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    marker=dict(size=10, opacity=0.4),
    name="Coaches"
))

# SIMILAR
similar = []
if active_coach in distance_df.index:
    similar = distance_df[active_coach].sort_values().drop(active_coach).head(5).index

sim_df = dna_df[dna_df["Coach"].isin(similar)]

fig.add_trace(go.Scatter(
    x=sim_df["xGF_60"],
    y=sim_df["xGA_60"],
    mode="markers",
    marker=dict(size=12, color="green"),
    text=sim_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="Similar"
))

# REPLACEMENT
replacement = coach_features.sort_values("xG_pct", ascending=False).head(5).index
rep_df = dna_df[dna_df["Coach"].isin(replacement)]

fig.add_trace(go.Scatter(
    x=rep_df["xGF_60"],
    y=rep_df["xGA_60"],
    mode="markers",
    marker=dict(size=12, color="blue"),
    text=rep_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="Replacement"
))

# ACTIVE
active_point = dna_df[dna_df["Coach"] == active_coach]

if not active_point.empty:
    fig.add_trace(go.Scatter(
        x=active_point["xGF_60"],
        y=active_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=[active_coach],
        textposition="top center",
        name="Active Coach"
    ))

fig.update_layout(
    hovermode="closest",
    legend=dict(orientation="h"),
    margin=dict(t=30)
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CONTROLS (INTERACTION ENGINE)
# =========================================================
col1, col2 = st.columns(2)

with col1:
    if st.button("Focus Sidebar Coach"):
        st.session_state.active_coach = selected_sidebar_coach

with col2:
    if st.button("Lock Focus"):
        st.session_state.active_coach = active_coach
