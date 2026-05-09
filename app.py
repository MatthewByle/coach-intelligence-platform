import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NHL Coaching",
    layout="wide"
)

st.title("NHL Coaching")

# -----------------------------
# GOOGLE SHEET
# -----------------------------
SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

# -----------------------------
# CLEAN DATA
# -----------------------------
# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
stats.columns = (
    stats.columns
    .str.replace("\u00a0", " ", regex=False)
    .str.strip()
)

coaches.columns = (
    coaches.columns
    .str.replace("\u00a0", " ", regex=False)
    .str.strip()
)

# Debug helper
# st.write(stats.columns)

# -----------------------------
# DATE CLEANING
# -----------------------------
if "Date" not in stats.columns:
    st.error(f"Date column not found. Available columns: {list(stats.columns)}")
    st.stop()

stats["Date"] = pd.to_datetime(
    stats["Date"],
    errors="coerce"
)

stats["xGF_60"] = pd.to_numeric(stats["xGF_60"], errors="coerce")
stats["xGA_60"] = pd.to_numeric(stats["xGA_60"], errors="coerce")
stats["xG_pct"] = pd.to_numeric(stats["xG_pct"], errors="coerce")
stats["PDO"] = pd.to_numeric(stats["PDO"], errors="coerce")

coaches["Head Coach"] = (
    coaches["Head Coach"]
    .astype(str)
    .str.strip()
)

coaches["Team Name"] = (
    coaches["Team Name"]
    .astype(str)
    .str.strip()
)

stats["Head Coach"] = (
    stats["Head Coach"]
    .astype(str)
    .str.strip()
)

stats["Team"] = (
    stats["Team"]
    .astype(str)
    .str.strip()
)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Navigation")

team_options = sorted(coaches["Team Name"].dropna().unique())
selected_team = st.sidebar.selectbox(
    "Select Team",
    ["All Teams"] + team_options
)

if selected_team == "All Teams":
    filtered_coaches = sorted(
        coaches["Head Coach"].dropna().unique()
    )
else:
    filtered_coaches = sorted(
        coaches[
            coaches["Team Name"] == selected_team
        ]["Head Coach"].dropna().unique()
    )

selected_coach = st.sidebar.selectbox(
    "Select Coach",
    filtered_coaches
)

# -----------------------------
# COACH LOOKUP
# -----------------------------
coach_filtered = coaches[
    coaches["Head Coach"] == selected_coach
]

if coach_filtered.empty:
    st.error("Coach not found.")
    st.stop()

coach_row = coach_filtered.iloc[0]

team = coach_row["Team Name"]
hire_date = pd.to_datetime(
    coach_row["Hire Date"],
    errors="coerce"
)

fire_date = pd.to_datetime(
    coach_row["Fire Date"],
    errors="coerce"
)

# -----------------------------
# TEAM DATA
# -----------------------------
team_data = stats[
    stats["Team"] == team
].copy()

team_data = team_data.sort_values("Date")

# -----------------------------
# BEFORE / AFTER ANALYSIS
# -----------------------------
before = team_data[
    team_data["Date"] < hire_date
].tail(25)

after = team_data[
    team_data["Date"] >= hire_date
].head(25)

if before.empty:
    before_xg = None
else:
    before_xg = float(before["xG_pct"].mean())

if after.empty:
    after_xg = None
else:
    after_xg = float(after["xG_pct"].mean())

if before_xg is not None and after_xg is not None:
    delta = after_xg - before_xg
else:
    delta = None

# -----------------------------
# TOP LAYOUT
# -----------------------------
left_col, right_col = st.columns([1.2, 2])

# -----------------------------
# COACH CARD
# -----------------------------
with left_col:

    st.subheader("Coach")

    st.markdown(f"## {selected_coach}")

    image_url = f"https://ui-avatars.com/api/?name={selected_coach.replace(' ', '+')}&size=300"

    img_col, text_col = st.columns([1, 2])

    with img_col:
        st.image(image_url, use_container_width=True)

    with text_col:
        st.markdown(f"### {team}")

    st.divider()

    st.subheader("Coach Scorecard")

    coach_games = stats[
        stats["Head Coach"] == selected_coach
    ]

    offense_score = (
        coach_games["xGF_60"].mean()
        / stats["xGF_60"].mean()
    ) * 100

    defense_score = (
        stats["xGA_60"].mean()
        / coach_games["xGA_60"].mean()
    ) * 100

    coach_score = (
        offense_score * 0.6
        + defense_score * 0.4
    )

    offense_score = round(offense_score, 1)
    defense_score = round(defense_score, 1)
    coach_score = round(coach_score, 1)

    if coach_score >= 110:
        grade = "A"
    elif coach_score >= 100:
        grade = "B"
    elif coach_score >= 90:
        grade = "C"
    elif coach_score >= 80:
        grade = "D"
    else:
        grade = "F"

    score_col1, score_col2 = st.columns(2)

    with score_col1:
        st.metric("Offense", offense_score)
        st.metric("Coach Score", coach_score)

    with score_col2:
        st.metric("Defense", defense_score)
        st.metric("Grade", grade)

# -----------------------------
# SYSTEM IMPACT
# -----------------------------
with right_col:

    st.subheader("System Impact")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "xG% Before",
        round(before_xg, 3) if before_xg else "N/A"
    )

    c2.metric(
        "xG% After",
        round(after_xg, 3) if after_xg else "N/A"
    )

    c3.metric(
        "Impact Delta",
        round(delta, 3) if delta is not None else "N/A"
    )

    st.divider()

    st.subheader("Team xG% Trend")

    chart_data = (
        team_data[["Date", "xG_pct"]]
        .dropna()
        .set_index("Date")
    )

    st.line_chart(chart_data)

# -----------------------------
# COACH DNA MODEL
# -----------------------------
coach_features = stats.groupby("Head Coach")[[
    "xGF_60",
    "xGA_60",
    "xG_pct",
    "PDO"
]].mean().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

coach_features["Cluster"] = kmeans.fit_predict(X_scaled)

distance_matrix = euclidean_distances(X_scaled)

distance_df = pd.DataFrame(
    distance_matrix,
    index=coach_features.index,
    columns=coach_features.index
)

dna_df = coach_features.reset_index()

# -----------------------------
# DNA MAP
# -----------------------------
st.subheader("Coach DNA Map")

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Head Coach",
    size="xG_pct",
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
        text=["Selected"],
        textposition="top center",
        name="Selected Coach"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# SIMILAR + REPLACEMENTS
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# MOST SIMILAR
# -----------------------------
with col1:

    st.subheader("Most Similar Coaches")

    similarities = (
        distance_df[selected_coach]
        .sort_values()
        .drop(selected_coach)
        .head(5)
    )

    similar_df = pd.DataFrame({
        "Coach": similarities.index
    })

    st.dataframe(
        similar_df,
        hide_index=True,
        use_container_width=True
    )

# -----------------------------
# REPLACEMENTS
# -----------------------------
with col2:

    st.subheader("Replacement Candidates")

    league_xg = stats["xG_pct"].mean()

    if after_xg is not None and after_xg < league_xg:

        better_fit = coach_features[
            coach_features["xG_pct"] > league_xg
        ]

        replacement_names = (
            better_fit.sort_values(
                "xG_pct",
                ascending=False
            )
            .head(5)
            .index
        )

    else:

        replacement_names = (
            coach_features
            .sort_values(
                "xGF_60",
                ascending=False
            )
            .head(5)
            .index
        )

    replacement_df = pd.DataFrame({
        "Coach": replacement_names
    })

    st.dataframe(
        replacement_df,
        hide_index=True,
        use_container_width=True
    )

# -----------------------------
# COACHING SUMMARY
# -----------------------------
st.subheader("Coaching Summary")

coach_profile = coach_features.loc[selected_coach]

offense = round(coach_profile["xGF_60"], 2)
defense = round(coach_profile["xGA_60"], 2)
xg = round(coach_profile["xG_pct"], 2)
pdo = round(coach_profile["PDO"], 3)

narrative = []

if offense > coach_features["xGF_60"].mean():
    narrative.append(
        "This coach drives above-average offensive creation."
    )
else:
    narrative.append(
        "This coach utilizes a more conservative offensive structure."
    )

if defense < coach_features["xGA_60"].mean():
    narrative.append(
        "Defensively, this system suppresses chances effectively."
    )
else:
    narrative.append(
        "The defensive profile allows elevated danger chances."
    )

if xg > coach_features["xG_pct"].mean():
    narrative.append(
        "Overall expected-goal control trends above league average."
    )
else:
    narrative.append(
        "Expected-goal control currently trends below league average."
    )

if pdo > 1.02:
    narrative.append(
        "PDO suggests results may be inflated by strong finishing or goaltending."
    )

if pdo < 0.98:
    narrative.append(
        "PDO suggests results may improve with regression."
    )

for sentence in narrative:
    st.write("•", sentence)

# -----------------------------
# TEAM STATS
# -----------------------------
st.subheader("Team Stats")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("xGF/60", offense)
metric2.metric("xGA/60", defense)
metric3.metric("xG%", xg)
metric4.metric("PDO", pdo)
