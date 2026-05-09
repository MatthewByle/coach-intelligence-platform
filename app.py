import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="NHL Coaching", layout="wide")
st.title("NHL Coaching")

# =========================================================
# SESSION STATE (CLICK-STYLE SELECTION SIMULATION)
# =========================================================
if "focus_coach" not in st.session_state:
    st.session_state.focus_coach = None

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
stats["Coach"] = stats["Coach"].astype(str).str.strip()

# =========================================================
# SIDEBAR
# =========================================================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

# Focus lock (click simulation override)
if st.session_state.focus_coach:
    active_coach = st.session_state.focus_coach
else:
    active_coach = selected_coach

# reset focus
if st.sidebar.button("Reset Focus"):
    st.session_state.focus_coach = None

# =========================================================
# MODEL
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

dna_df = coach_features.reset_index()

# =========================================================
# DNA MAP (SCOUTING-GRADE INTERACTIVE MODE)
# =========================================================
st.subheader("Coach DNA Map (Scouting Mode)")

fig = go.Figure()

# -------------------------
# BASE LAYER
# -------------------------
fig.add_trace(go.Scatter(
    x=dna_df["xGF_60"],
    y=dna_df["xGA_60"],
    mode="markers",
    text=dna_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    marker=dict(size=10, opacity=0.5),
    name="Coaches"
))

# -------------------------
# SIMILAR COACHES
# -------------------------
similar_list = []
if active_coach in distance_df.index:
    similar_list = (
        distance_df[active_coach]
        .sort_values()
        .drop(active_coach)
        .head(5)
        .index
    )

sim_df = dna_df[dna_df["Coach"].isin(similar_list)]

fig.add_trace(go.Scatter(
    x=sim_df["xGF_60"],
    y=sim_df["xGA_60"],
    mode="markers",
    marker=dict(size=12, color="green"),
    name="Similar Coaches",
    hovertemplate="<b>%{text}</b><extra></extra>",
    text=sim_df["Coach"]
))

# -------------------------
# REPLACEMENT COACHES
# -------------------------
replacement_list = coach_features.sort_values("xG_pct", ascending=False).head(5).index
rep_df = dna_df[dna_df["Coach"].isin(replacement_list)]

fig.add_trace(go.Scatter(
    x=rep_df["xGF_60"],
    y=rep_df["xGA_60"],
    mode="markers",
    marker=dict(size=12, color="blue"),
    name="Replacement Candidates",
    hovertemplate="<b>%{text}</b><extra></extra>",
    text=rep_df["Coach"]
))

# -------------------------
# SELECTED / FOCUS COACH (RED + BIG)
# -------------------------
focus_point = dna_df[dna_df["Coach"] == active_coach]

if not focus_point.empty:
    fig.add_trace(go.Scatter(
        x=focus_point["xGF_60"],
        y=focus_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=20, color="red"),
        text=[active_coach],
        textposition="top center",
        name="Focused Coach"
    ))

# -------------------------
# INTERACTIVE LAYOUT
# -------------------------
fig.update_layout(
    hovermode="closest",
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(t=40)
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CLICK SIMULATION CONTROLS (SCOUTING MODE)
# =========================================================
st.caption("Click simulation: select a coach to focus")

col1, col2 = st.columns(2)

with col1:
    if st.button("Focus Selected Coach"):
        st.session_state.focus_coach = selected_coach

with col2:
    if st.button("Focus Similar Top Coach"):
        if len(similar_list) > 0:
            st.session_state.focus_coach = list(similar_list)[0]

# =========================================================
# QUICK CONTEXT PANEL
# =========================================================
st.subheader("Focused Coach Context")

st.write(f"**Active Focus:** {active_coach}")
st.write(f"Similar Pool: {len(similar_list)} coaches")
st.write(f"Replacement Pool: {len(replacement_list)} coaches")
