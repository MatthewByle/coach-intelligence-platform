import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# =========================
# CONFIG
# =========================
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

st.set_page_config(layout="wide")
st.title("NHL Head Coach Value")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# =========================
# CLEAN DATA (CRITICAL FIX)
# =========================
def clean(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace("\u00a0", " ").str.strip()
    return df

stats = clean(stats)
coaches = clean(coaches)

stats.columns = stats.columns.str.strip()
stats = stats.rename(columns={"Coach": "Head Coach"})

# Convert types
stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")
stats["xG_pct"] = pd.to_numeric(stats["xG_pct"], errors="coerce")

# =========================
# SIDEBAR COACH SELECT
# =========================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

# =========================
# COACH LOOKUP
# =========================
filtered = coaches[coaches["Head Coach"] == selected_coach]

if filtered.empty:
    st.error("Coach not found.")
    st.stop()

coach_row = filtered.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(coach_row["Hire Date"], errors="coerce")

# =========================
# CONTEXT
# =========================
st.subheader("Coach Context")
st.write(f"Team: {team}")
st.write(f"Hire Date: {hire_date}")

# =========================
# TEAM DATA
# =========================
team_data = stats[stats["Team"] == team].copy()

before = team_data[team_data["Date"] < hire_date].tail(15)
after = team_data[team_data["Date"] >= hire_date].head(15)

def safe_mean(df, col):
    return df[col].mean() if not df.empty else None

before_xg = safe_mean(before, "xG_pct")
after_xg = safe_mean(after, "xG_pct")

delta = (
    (after_xg - before_xg)
    if before_xg is not None and after_xg is not None
    else None
)

# =========================
# BEFORE / AFTER UI
# =========================
st.subheader("System Impact")

col1, col2, col3 = st.columns(3)

col1.metric("xG% Before", f"{before_xg:.3f}" if before_xg else "N/A")
col2.metric("xG% After", f"{after_xg:.3f}" if after_xg else "N/A")
col3.metric("Impact Delta", f"{delta:.3f}" if delta is not None else "N/A")

# =========================
# SCORECARD
# =========================
team_stats = stats[stats["Team"] == team]

offense_score = team_stats["xGF_60"].rank(pct=True).mean() * 100
defense_score = (1 - team_stats["xGA_60"].rank(pct=True)).mean() * 100
coach_score = (0.6 * offense_score) + (0.4 * defense_score)

st.subheader("Coach Scorecard")

st.metric("Offense", round(offense_score, 1))
st.metric("Defense", round(defense_score, 1))
st.metric("Overall", round(coach_score, 1))

# =========================
# FEATURE MATRIX (DNA)
# =========================
coach_features = stats.groupby("Head Coach")[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]].mean().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
coach_features["Cluster"] = kmeans.fit_predict(X_scaled)

dna_df = coach_features.reset_index()

# =========================
# SIMILARITY ENGINE (COACH-TO-COACH)
# =========================
distance_matrix = euclidean_distances(X_scaled)

distance_df = pd.DataFrame(
    distance_matrix,
    index=coach_features.index,
    columns=coach_features.index
)

st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:
    sims = distance_df[selected_coach].sort_values().drop(selected_coach).head(5)

    st.dataframe(pd.DataFrame({
        "Coach": sims.index,
        "Similarity": sims.values.round(2)
    }))

# =========================
# REPLACEMENT ENGINE (TEAM NEEDS MODEL) ⭐ NEW
# =========================
st.subheader("Replacement Candidates (Team Needs Model)")

league_avg = stats[[
    "xGF_60",
    "xGA_60",
    "xG_pct"
]].mean()

team_profile = team_data[[
    "xGF_60",
    "xGA_60",
    "xG_pct"
]].mean()

team_needs = pd.Series({
    "offense_need": league_avg["xGF_60"] - team_profile["xGF_60"],
    "defense_need": team_profile["xGA_60"] - league_avg["xGA_60"],
    "structure_need": league_avg["xG_pct"] - team_profile["xG_pct"]
})

replacement_scores = coach_features.copy()

replacement_scores["offense_strength"] = replacement_scores["xGF_60"] - league_avg["xGF_60"]
replacement_scores["defense_strength"] = league_avg["xGA_60"] - replacement_scores["xGA_60"]
replacement_scores["structure_strength"] = replacement_scores["xG_pct"] - league_avg["xG_pct"]

replacement_scores["Fit Score"] = (
    team_needs["offense_need"] * replacement_scores["offense_strength"] +
    team_needs["defense_need"] * replacement_scores["defense_strength"] +
    team_needs["structure_need"] * replacement_scores["structure_strength"]
)

top_replacements = replacement_scores.sort_values("Fit Score", ascending=False).head(5)

st.dataframe(top_replacements.reset_index()[[
    "Head Coach",
    "Fit Score",
    "xGF_60",
    "xGA_60",
    "xG_pct"
]])

# =========================
# DNA MAP
# =========================
st.subheader("Coach DNA Map")

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Head Coach",
    size="xG_pct"
)

selected_row = dna_df[dna_df["Head Coach"] == selected_coach]

if not selected_row.empty:
    fig.add_scatter(
        x=selected_row["xGF_60"],
        y=selected_row["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=["YOU"],
        name="Selected Coach"
    )

st.plotly_chart(fig, use_container_width=True)

# =========================
# NARRATIVE LAYER
# =========================
st.subheader("Coach Narrative")

profile = coach_features.loc[selected_coach]

narrative = []

if profile["xGF_60"] > coach_features["xGF_60"].mean():
    narrative.append("Offensive-oriented system with strong chance creation.")
else:
    narrative.append("Controlled offensive structure.")

if profile["xGA_60"] < coach_features["xGA_60"].mean():
    narrative.append("Defensive stability above league average.")
else:
    narrative.append("Defensive volatility present.")

for n in narrative:
    st.write("•", n)

# =========================
# PROFILE CARD
# =========================
st.subheader("Coach Profile Card")

row = dna_df[dna_df["Head Coach"] == selected_coach].iloc[0]

st.markdown(f"""
### {selected_coach}

**Cluster:** {row["Cluster"]}

- xGF/60: {row["xGF_60"]:.2f}
- xGA/60: {row["xGA_60"]:.2f}
- xG%: {row["xG_pct"]:.2f}
- PDO: {row["PDO"]:.3f}
""")
