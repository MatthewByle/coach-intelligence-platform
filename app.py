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
# CLEAN DATA
# =========================
def clean(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u00a0", " ")
                .str.strip()
            )

    return df

stats = clean(stats)
coaches = clean(coaches)

stats.columns = stats.columns.str.strip()
coaches.columns = coaches.columns.str.strip()

stats = stats.rename(columns={"Coach": "Head Coach"})

# =========================
# TYPE CONVERSION
# =========================
stats["Date"] = pd.to_datetime(stats["Date"], errors="coerce")
stats["xG_pct"] = pd.to_numeric(stats["xG_pct"], errors="coerce")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

team_options = sorted(coaches["Team Name"].dropna().unique())
selected_team = st.sidebar.selectbox(
    "Select Team",
    ["All Teams"] + team_options
)

if selected_team != "All Teams":
    filtered_coaches = coaches[
        coaches["Team Name"] == selected_team
    ]
else:
    filtered_coaches = coaches

coach_list = sorted(
    filtered_coaches["Head Coach"].dropna().unique()
)

selected_coach = st.sidebar.selectbox(
    "Select Coach",
    coach_list
)

# =========================
# COACH LOOKUP
# =========================
filtered = coaches[
    coaches["Head Coach"] == selected_coach
]

if filtered.empty:
    st.error("Coach not found.")
    st.stop()

coach_row = filtered.iloc[0]

team = coach_row["Team Name"]

# Optional image column
coach_image = (
    coach_row["Image"]
    if "Image" in coaches.columns
    else None
)

hire_date = pd.to_datetime(
    coach_row["Hire Date"],
    errors="coerce"
)

# =========================
# TEAM DATA
# =========================
team_data = stats[
    stats["Team"] == team
].copy()

before = team_data[
    team_data["Date"] < hire_date
].tail(15)

after = team_data[
    team_data["Date"] >= hire_date
].head(15)

# =========================
# SAFE MEAN FUNCTION
# =========================
def safe_mean(df, col):
    return df[col].mean() if not df.empty else None

before_xg = safe_mean(before, "xG_pct")
after_xg = safe_mean(after, "xG_pct")

delta = (
    after_xg - before_xg
    if before_xg is not None and after_xg is not None
    else None
)

# =========================
# TOP SECTION LAYOUT
# =========================
left_col, right_col = st.columns([2, 1])

# =========================
# LEFT SIDE
# =========================
with left_col:

    st.subheader("System Impact")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "xG% Before",
        f"{before_xg:.3f}" if before_xg is not None else "N/A"
    )

    col2.metric(
        "xG% After",
        f"{after_xg:.3f}" if after_xg is not None else "N/A"
    )

    col3.metric(
        "Impact Delta",
        f"{delta:.3f}" if delta is not None else "N/A"
    )

    st.subheader("Team xG% Trend")

    if not team_data.empty:
        st.line_chart(
            team_data[["Date", "xG_pct"]]
            .set_index("Date")
        )

# =========================
# RIGHT SIDE
# =========================
with right_col:

    st.subheader("Coach")

    st.markdown(f"### {selected_coach}")

    image_col, info_col = st.columns([1, 2])

    with image_col:

        if coach_image and coach_image != "nan":
            st.image(coach_image, width=120)
        else:
            st.image(
                "https://via.placeholder.com/120x120.png?text=Coach",
                width=120
            )

    with info_col:

        st.write(f"**Team:** {team}")

# =========================
# SCORECARD
# =========================
coach_games = stats[
    stats["Head Coach"] == selected_coach
]

offense_score = (
    coach_games["xGF_60"]
    .rank(pct=True)
    .mean() * 100
)

defense_score = (
    1 - coach_games["xGA_60"]
    .rank(pct=True)
).mean() * 100

coach_score = (
    (0.6 * offense_score) +
    (0.4 * defense_score)
)

st.subheader("Coach Scorecard")

s1, s2, s3 = st.columns(3)

s1.metric("Offense", round(offense_score, 1))
s2.metric("Defense", round(defense_score, 1))
s3.metric("Overall", round(coach_score, 1))

# =========================
# COACH FEATURE MATRIX
# =========================
coach_features = stats.groupby("Head Coach")[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]].mean().dropna()

# =========================
# CLUSTERING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

coach_features["Cluster"] = (
    kmeans.fit_predict(X_scaled)
)

dna_df = coach_features.reset_index()

# =========================
# SIMILARITY ENGINE
# =========================
distance_matrix = euclidean_distances(X_scaled)

distance_df = pd.DataFrame(
    distance_matrix,
    index=coach_features.index,
    columns=coach_features.index
)

st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:

    sims = (
        distance_df[selected_coach]
        .sort_values()
        .drop(selected_coach)
        .head(5)
    )

    similar_df = pd.DataFrame({
        "Coach": sims.index,
        "Similarity": sims.values.round(2)
    })

    similar_df = similar_df.reset_index(drop=True)

    st.dataframe(
        similar_df,
        use_container_width=True
    )

# =========================
# REPLACEMENT ENGINE
# =========================
st.subheader("Replacement Candidates")

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
    "offense_need":
        league_avg["xGF_60"] - team_profile["xGF_60"],

    "defense_need":
        team_profile["xGA_60"] - league_avg["xGA_60"],

    "structure_need":
        league_avg["xG_pct"] - team_profile["xG_pct"]
})

replacement_scores = coach_features.copy()

replacement_scores["offense_strength"] = (
    replacement_scores["xGF_60"] -
    league_avg["xGF_60"]
)

replacement_scores["defense_strength"] = (
    league_avg["xGA_60"] -
    replacement_scores["xGA_60"]
)

replacement_scores["structure_strength"] = (
    replacement_scores["xG_pct"] -
    league_avg["xG_pct"]
)

replacement_scores["Fit Score"] = (
    team_needs["offense_need"] *
    replacement_scores["offense_strength"] +

    team_needs["defense_need"] *
    replacement_scores["defense_strength"] +

    team_needs["structure_need"] *
    replacement_scores["structure_strength"]
)

top_replacements = (
    replacement_scores
    .sort_values("Fit Score", ascending=False)
    .head(5)
)

replacement_df = (
    top_replacements
    .reset_index()[[
        "Head Coach",
        "Fit Score",
        "xGF_60",
        "xGA_60",
        "xG_pct"
    ]]
)

replacement_df = replacement_df.reset_index(drop=True)

st.dataframe(
    replacement_df,
    use_container_width=True
)

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

selected_row = dna_df[
    dna_df["Head Coach"] == selected_coach
]

if not selected_row.empty:

    fig.add_scatter(
        x=selected_row["xGF_60"],
        y=selected_row["xGA_60"],
        mode="markers+text",
        marker=dict(
            size=18,
            color="red"
        ),
        text=["YOU"],
        textposition="top center",
        name="Selected Coach"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# COACHING SUMMARY
# =========================
st.subheader("Coaching Summary")

profile = coach_features.loc[selected_coach]

narrative = []

if profile["xGF_60"] > coach_features["xGF_60"].mean():

    narrative.append(
        "Offensive-oriented system with strong chance creation."
    )

else:

    narrative.append(
        "More controlled offensive structure."
    )

if profile["xGA_60"] < coach_features["xGA_60"].mean():

    narrative.append(
        "Defensive suppression trends above league average."
    )

else:

    narrative.append(
        "Defensive structure allows elevated chances against."
    )

if profile["xG_pct"] > coach_features["xG_pct"].mean():

    narrative.append(
        "Expected-goal control trends positively overall."
    )

else:

    narrative.append(
        "Expected-goal control trends below league average."
    )

for line in narrative:
    st.write("•", line)

# =========================
# TEAM STATS
# =========================
st.subheader("Team Stats")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

stats_col1.metric(
    "xGF/60",
    round(profile["xGF_60"], 2)
)

stats_col2.metric(
    "xGA/60",
    round(profile["xGA_60"], 2)
)

stats_col3.metric(
    "xG%",
    round(profile["xG_pct"], 2)
)

stats_col4.metric(
    "PDO",
    round(profile["PDO"], 3)
)
