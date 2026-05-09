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
st.title("Coach Intelligence Dashboard")

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
# CLEAN DATA (CRITICAL)
# =========================
def clean_strings(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace("\u00a0", " ").str.strip()
    return df

coaches = clean_strings(coaches)
stats = clean_strings(stats)

stats.columns = stats.columns.str.strip()

# Standardize naming mismatch
stats = stats.rename(columns={"Coach": "Head Coach"})

# Convert types safely
stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")
stats["xG_pct"] = pd.to_numeric(stats["xG_pct"], errors="coerce")

# =========================
# SIDEBAR INPUT
# =========================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

# =========================
# COACH LOOKUP
# =========================
filtered = coaches[coaches["Head Coach"] == selected_coach]

if filtered.empty:
    st.error("Coach not found in dataset.")
    st.stop()

coach_row = filtered.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(coach_row["Hire Date"], errors="coerce")
fire_date = pd.to_datetime(coach_row["Fire Date"], errors="coerce")

# =========================
# COACH CONTEXT UI
# =========================
st.subheader("Coach Context")
st.write(f"**Team:** {team}")
st.write(f"**Hire Date:** {hire_date}")
st.write(f"**Fire Date:** {fire_date}")

# =========================
# TEAM FILTER
# =========================
team_data = stats[stats["Team"] == team].copy()

before = team_data[team_data["Date"] < hire_date].tail(15)
after = team_data[team_data["Date"] >= hire_date].head(15)

team_profile = team_data[[
    "xGF_60",
    "xGA_60",
    "xG_pct"
]].mean()

league_avg = stats[[
    "xGF_60",
    "xGA_60",
    "xG_pct"
]].mean()

team_needs = pd.Series({
    "offense_need": league_avg["xGF_60"] - team_profile["xGF_60"],
    "defense_need": team_profile["xGA_60"] - league_avg["xGA_60"],
    "structure_need": league_avg["xG_pct"] - team_profile["xG_pct"]
})

# =========================
# BEFORE / AFTER ANALYSIS
# =========================
def safe_mean(df, col):
    return df[col].mean() if not df.empty else None

before_xg = safe_mean(before, "xG_pct")
after_xg = safe_mean(after, "xG_pct")

delta = (
    (after_xg - before_xg)
    if before_xg is not None and after_xg is not None
    else None
)

st.subheader("System Impact (Before vs After)")

col1, col2, col3 = st.columns(3)

col1.metric("xG% Before", f"{before_xg:.3f}" if before_xg is not None else "N/A")
col2.metric("xG% After", f"{after_xg:.3f}" if after_xg is not None else "N/A")
col3.metric("Impact Delta", f"{delta:.3f}" if delta is not None else "N/A")

# =========================
# TREND CHART
# =========================
st.subheader("Team xG% Trend")

if not team_data.empty:
    st.line_chart(team_data[["Date", "xG_pct"]].set_index("Date"))

# =========================
# COACH PERFORMANCE SCORE
# =========================
team_stats = stats[stats["Team"] == team]

offense_score = team_stats["xGF_60"].rank(pct=True).mean() * 100
defense_score = (1 - team_stats["xGA_60"].rank(pct=True)).mean() * 100
coach_score = (0.6 * offense_score) + (0.4 * defense_score)

grade = (
    "A" if coach_score >= 80 else
    "B" if coach_score >= 70 else
    "C" if coach_score >= 60 else
    "D" if coach_score >= 50 else
    "F"
)

st.subheader("Coach Scorecard")

st.metric("Offense", round(offense_score, 1))
st.metric("Defense", round(defense_score, 1))
st.metric("Overall Score", round(coach_score, 1))
st.success(f"Grade: {grade}")

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
# SIMILARITY ENGINE
# =========================
st.subheader("Replacement Candidates (Team Fit Model)")

replacements = replacement_scores.sort_values("Fit Score", ascending=False).head(5)

replacement_df = replacements.reset_index()[[
    "Head Coach",
    "Fit Score",
    "xGF_60",
    "xGA_60",
    "xG_pct"
]]

st.dataframe(replacement_df)

# =========================
# REPLACEMENT ENGINE
# =========================
replacement_scores = coach_features.copy()

# Normalize coach strengths relative to league
replacement_scores["offense_strength"] = replacement_scores["xGF_60"] - league_avg["xGF_60"]
replacement_scores["defense_strength"] = league_avg["xGA_60"] - replacement_scores["xGA_60"]
replacement_scores["structure_strength"] = replacement_scores["xG_pct"] - league_avg["xG_pct"]

# Fit calculation (TEAM NEED MATCHING)
replacement_scores["Fit Score"] = (
    team_needs["offense_need"] * replacement_scores["offense_strength"] +
    team_needs["defense_need"] * replacement_scores["defense_strength"] +
    team_needs["structure_need"] * replacement_scores["structure_strength"]
)

# =========================
# NARRATIVE LAYER
# =========================
st.subheader("Coach Narrative")

coach_profile = coach_features.loc[selected_coach]

narrative = []

if coach_profile["xGF_60"] > coach_features["xGF_60"].mean():
    narrative.append("Offensive-heavy system with strong chance creation.")
else:
    narrative.append("More structured and controlled offensive system.")

if coach_profile["xGA_60"] < coach_features["xGA_60"].mean():
    narrative.append("Strong defensive suppression profile.")
else:
    narrative.append("Allows higher defensive volatility.")

if coach_profile["xG_pct"] > coach_features["xG_pct"].mean():
    narrative.append("Above-average expected goal control.")
else:
    narrative.append("Below-average expected goal control.")

for line in narrative:
    st.write("•", line)

# =========================
# COACH PROFILE CARD
# =========================
def get_archetype(cluster):
    return {
        0: "Balanced System",
        1: "Offensive Pressure",
        2: "Defensive Structure",
        3: "High Variance"
    }.get(cluster, "Unknown")

coach_profile_row = dna_df[dna_df["Head Coach"] == selected_coach].iloc[0]
archetype = get_archetype(coach_profile_row["Cluster"])

st.subheader("Coach Profile")

st.markdown(f"""
### {selected_coach}

**Archetype:** {archetype}

- xGF/60: {coach_profile_row["xGF_60"]:.2f}
- xGA/60: {coach_profile_row["xGA_60"]:.2f}
- xG%: {coach_profile_row["xG_pct"]:.2f}
- PDO: {coach_profile_row["PDO"]:.3f}

System Identity: {archetype} profile based on clustering model.
""")
