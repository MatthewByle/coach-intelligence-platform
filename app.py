import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NHL Coaching",
    layout="wide"
)

st.title("NHL Coaching")

# =========================================================
# LOAD DATA
# =========================================================
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet):
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet}"
    )
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

# Numeric columns
numeric_cols = [
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]

for col in numeric_cols:
    if col in stats.columns:
        stats[col] = pd.to_numeric(
            stats[col],
            errors="coerce"
        )

# Date handling
if "Date" in stats.columns:
    stats["Date"] = pd.to_datetime(
        stats["Date"],
        errors="coerce"
    )

stats["Coach"] = (
    stats["Coach"]
    .astype(str)
    .str.strip()
)

coaches["Head Coach"] = (
    coaches["Head Coach"]
    .astype(str)
    .str.strip()
)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.header("Navigation")

# -------------------------
# TEAM DROPDOWN
# -------------------------
team_list = sorted(
    coaches["Team Name"]
    .dropna()
    .unique()
)

selected_team = st.sidebar.selectbox(
    "Select Team",
    team_list
)

# -------------------------
# FILTER COACHES BY TEAM
# -------------------------
filtered_coaches = coaches[
    coaches["Team Name"] == selected_team
]

coach_list = sorted(
    filtered_coaches["Head Coach"]
    .dropna()
    .unique()
)

# -------------------------
# COACH DROPDOWN
# -------------------------
selected_coach = st.sidebar.selectbox(
    "Select Coach",
    coach_list
)

# =========================================================
# COACH LOOKUP
# =========================================================
coach_row = coaches[
    coaches["Head Coach"] == selected_coach
].iloc[0]

team = coach_row["Team Name"]

coach_games = stats[
    stats["Coach"] == selected_coach
].copy()

team_stats = stats[
    stats["Team"] == team
].copy()

# =========================================================
# COACH HEADER
# =========================================================
st.subheader("Coach")

header_col1, header_col2 = st.columns([1, 4])

with header_col1:

    # Optional coach image support
    if "Image" in coaches.columns:
        image_url = coach_row.get("Image", None)

        if pd.notna(image_url):
            st.image(image_url, width=140)

with header_col2:
    st.markdown(f"## {selected_coach}")
    st.markdown(f"### {team}")

# =========================================================
# DNA MODEL
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]].mean()

coach_features = coach_features.fillna(
    coach_features.mean()
)

coach_features.index = (
    coach_features.index
    .astype(str)
    .str.strip()
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

coach_features["Cluster"] = kmeans.fit_predict(X_scaled)

dna_df = coach_features.reset_index()

# =========================================================
# ROLE SYSTEM
# =========================================================
def get_role(cluster):

    mapping = {
        0: "Balanced System Coach",
        1: "Offensive Pressure Coach",
        2: "Defensive Structure Coach",
        3: "Transition / High-Variance Coach"
    }

    return mapping.get(cluster, "Unknown")

dna_df["Role"] = dna_df["Cluster"].apply(get_role)

# =========================================================
# SAFE ROLE LOOKUP
# =========================================================

role_lookup = dna_df[
    dna_df["Coach"]
    .astype(str)
    .str.strip()
    ==
    str(selected_coach).strip()
]

if not role_lookup.empty:
    coach_role = role_lookup["Role"].iloc[0]
else:
    coach_role = "Unknown"

# =========================================================
# SIMILARITY ENGINE
# =========================================================
distance_df = pd.DataFrame(
    euclidean_distances(X_scaled),
    index=coach_features.index,
    columns=coach_features.index
)

# =========================================================
# SAFE SIMILARITY LOOKUP
# =========================================================

selected_clean = str(selected_coach).strip()

distance_df.columns = (
    distance_df.columns.astype(str).str.strip()
)

distance_df.index = (
    distance_df.index.astype(str).str.strip()
)

if selected_clean in distance_df.columns:

    similarities = (
        distance_df[selected_clean]
        .sort_values()
        .drop(selected_clean)
        .head(5)
    )

    similar_coaches = similarities.index.tolist()

else:

    st.warning(
        f"No similarity profile found for {selected_clean}"
    )

    similarities = pd.Series(dtype=float)
    similar_coaches = []

# =========================================================
# REPLACEMENT ENGINE
# =========================================================
# =========================================================
# SAFE CLUSTER LOOKUP
# =========================================================

cluster_lookup = dna_df[
    dna_df["Coach"]
    .astype(str)
    .str.strip()
    ==
    str(selected_coach).strip()
]

if not cluster_lookup.empty:

    selected_cluster = (
        cluster_lookup["Cluster"]
        .iloc[0]
    )

else:

    st.warning(
        f"No cluster profile found for {selected_coach}"
    )

    selected_cluster = None

if selected_cluster is not None:

    replacement_df = dna_df[
        (dna_df["Cluster"] == selected_cluster)
        &
        (
            dna_df["Coach"]
            .astype(str)
            .str.strip()
            !=
            str(selected_coach).strip()
        )
    ].copy()

else:

    replacement_df = pd.DataFrame()

# =========================================================
# SAFE REPLACEMENT CANDIDATES
# =========================================================

if not replacement_df.empty and "Coach" in replacement_df.columns:

    replacement_candidates = (
        replacement_df["Coach"]
        .head(5)
        .tolist()
    )

else:

    replacement_candidates = []

# =========================================================
# SCORECARD
# =========================================================
st.subheader("Coach Scorecard")

league = stats.copy()

def percentile(series, value):
    return (series <= value).mean()

off_pct = percentile(
    league["xGF_60"].dropna(),
    coach_games["xGF_60"].mean()
)

def_pct = percentile(
    league["xGA_60"].dropna(),
    coach_games["xGA_60"].mean()
)

sys_pct = percentile(
    league["xG_pct"].dropna(),
    coach_games["xG_pct"].mean()
)

def to_grade(p):

    if p >= 0.90:
        return "A"

    elif p >= 0.75:
        return "B"

    elif p >= 0.50:
        return "C"

    elif p >= 0.30:
        return "D"

    else:
        return "F"

off_grade = to_grade(off_pct)
def_grade = to_grade(1 - def_pct)
sys_grade = to_grade(sys_pct)

grade_map = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "F": 1
}

overall_num = (
    grade_map[off_grade]
    + grade_map[def_grade]
    + grade_map[sys_grade]
) / 3

if overall_num >= 4.5:
    overall_grade = "A"
elif overall_num >= 3.5:
    overall_grade = "B"
elif overall_num >= 2.5:
    overall_grade = "C"
elif overall_num >= 1.5:
    overall_grade = "D"
else:
    overall_grade = "F"

c1, c2, c3, c4 = st.columns(4)

c1.metric("Offense", off_grade)
c2.metric("Defense", def_grade)
c3.metric("System", sys_grade)
c4.metric("Overall", overall_grade)

# =========================================================
# AI COACHING NARRATIVE
# =========================================================
st.subheader("AI Coaching Narrative")

st.write(f"• Coaching Archetype: {coach_role}")

if coach_games["xGF_60"].mean() > stats["xGF_60"].mean():
    st.write(
        "• Offensive systems generate chances above league average."
    )
else:
    st.write(
        "• Offensive generation relies more on structure and efficiency."
    )

if coach_games["xGA_60"].mean() < stats["xGA_60"].mean():
    st.write(
        "• Defensive suppression trends positively against league baseline."
    )
else:
    st.write(
        "• Defensive environment can become high-event under pressure."
    )

if coach_games["xG_pct"].mean() > stats["xG_pct"].mean():
    st.write(
        "• Teams consistently control expected-goal share."
    )
else:
    st.write(
        "• Expected-goal control remains inconsistent."
    )

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

team_profile = team_stats[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]].mean()

t1, t2, t3, t4 = st.columns(4)

t1.metric("xGF/60", round(team_profile["xGF_60"], 2))
t2.metric("xGA/60", round(team_profile["xGA_60"], 2))
t3.metric("xG%", round(team_profile["xG_pct"], 2))
t4.metric("PDO", round(team_profile["PDO"], 3))

# =========================================================
# SYSTEM IMPACT
# =========================================================
st.subheader("System Impact")

if "Date" in stats.columns:

    team_stats = team_stats.sort_values("Date")

    split = len(team_stats) // 2

    before = team_stats.iloc[:split]
    after = team_stats.iloc[split:]

    before_xg = before["xG_pct"].mean()
    after_xg = after["xG_pct"].mean()

    delta = after_xg - before_xg

s1, s2, s3 = st.columns(3)

s1.metric(
    "Before",
    round(before_xg, 2)
)

s2.metric(
    "After",
    round(after_xg, 2)
)

impact_color = "normal"

if delta < 0:
    impact_color = "inverse"

s3.metric(
    "Impact",
    f"{delta:+.2f}",
    delta_color=impact_color
)

# =========================================================
# TEAM TREND
# =========================================================
if "Date" in stats.columns:

    st.subheader("Team xG% Trend")

    chart_data = (
        team_stats[["Date", "xG_pct"]]
        .dropna()
        .set_index("Date")
    )

    st.line_chart(chart_data)

# =========================================================
# DNA MAP
# =========================================================
st.subheader("Coach DNA Map")

dna_df["Type"] = "League Coach"

dna_df.loc[
    dna_df["Coach"] == selected_coach,
    "Type"
] = "Selected Coach"

dna_df.loc[
    dna_df["Coach"].isin(similar_coaches),
    "Type"
] = "Similar Coach"

dna_df.loc[
    dna_df["Coach"].isin(replacement_candidates),
    "Type"
] = "Replacement Candidate"

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Type",
    size="xG_pct",
    hover_name="Coach",
    hover_data={
        "xGF_60": True,
        "xGA_60": True,
        "xG_pct": True,
        "PDO": True,
        "Cluster": False,
        "Type": True
    }
)

fig.update_traces(
    marker=dict(
        sizemode="diameter",
        line=dict(width=1)
    ),
    selector=dict(mode="markers")
)

fig.update_layout(
    height=700,
    hovermode="closest",
    legend_title_text=""
)

# Hover growth effect
fig.update_traces(
    marker=dict(size=14),
    selected=dict(
        marker=dict(size=22)
    ),
    unselected=dict(
        marker=dict(opacity=0.75)
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# MOST SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

similar_df = pd.DataFrame({
    "Coach": similarities.index,
    "Similarity Score": similarities.values.round(2)
})

st.dataframe(
    similar_df,
    hide_index=True,
    use_container_width=True
)

# =========================================================
# REPLACEMENT CANDIDATES
# =========================================================
st.subheader("Replacement Candidates")

replacement_display = replacement_df[[
    "Coach",
    "Role",
    "xGF_60",
    "xGA_60",
    "xG_pct"
]].copy()

replacement_display.columns = [
    "Coach",
    "Archetype",
    "xGF/60",
    "xGA/60",
    "xG%",
]

st.dataframe(
    replacement_display,
    hide_index=True,
    use_container_width=True
)
