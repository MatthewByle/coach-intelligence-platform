import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

stats.columns = stats.columns.str.strip()

stats = stats.rename(columns={"Coach": "Head Coach"})

stats.columns = stats.columns.str.strip()

st.title("Coach Intelligence Dashboard")

st.write("Coach Registry Columns:")
st.write(coaches.columns)

st.write("Coach Registry Preview:")
st.dataframe(coaches.head())

coach_list = coaches["Head Coach"].dropna().unique()
selected_coach = st.sidebar.selectbox("Select Coach", coach_list)

st.write("Selected Coach:", selected_coach)

coach_row = coaches[coaches["Head Coach"] == selected_coach].iloc[0]

team = coach_row["Team Name"]
hire_date = coach_row["Hire Date"]
fire_date = coach_row["Fire Date"]

st.subheader("Coach Context")
st.write("Team:", team)
st.write("Hire Date:", hire_date)
st.write("Fire Date:", fire_date)

st.subheader("Coach Context")
st.write("Team:", team)
st.write("Hire Date:", hire_date)
st.write("Fire Date:", fire_date)

stats["Date"] = pd.to_datetime(stats["Date"])

team_data = stats[stats["Team"] == team].copy()
team_data = team_data.sort_values("Date")

hire_date = pd.to_datetime(hire_date)

before = team_data[team_data["Date"] < hire_date].tail(15)
after = team_data[team_data["Date"] >= hire_date].head(15)

before_xg = before["xG_pct"].mean()
after_xg = after["xG_pct"].mean()

delta = after_xg - before_xg

st.subheader("System Impact (Before vs After)")

st.metric("xG% Before", round(before_xg, 3))
st.metric("xG% After", round(after_xg, 3))
st.metric("Impact Delta", round(delta, 3))

stats.columns = stats.columns.str.strip()

st.subheader("Coach Performance Summary")

if delta > 0:
    st.success("System improved under this coach")
elif delta < 0:
    st.error("System declined under this coach")
else:
    st.info("No measurable change")

st.write("Sample sizes:")
st.write("Before games:", len(before))
st.write("After games:", len(after))

st.subheader("Team xG% Trend (Season)")

st.line_chart(
    team_data[["Date", "xG_pct"]].set_index("Date")
)

offense_score = stats["xGF_60"].rank(pct=True).mean() * 100

defense_score = (1 - stats["xGA_60"].rank(pct=True)).mean() * 100

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

st.subheader("Coach Performance Scorecard")

st.metric("Offense Score", round(offense_score, 1))
st.metric("Defense Score", round(defense_score, 1))
st.metric("Coach Score", round(coach_score, 1))

st.success(f"Coach Grade: {grade}")

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

coach_cluster_map = coach_features["Cluster"]

st.subheader("Coach DNA Map (Clustering)")

if selected_coach in coach_features.index:
    selected_cluster = coach_features.loc[selected_coach, "Cluster"]
else:
    st.error("Coach not found in DNA model")
    selected_cluster = None

st.write(f"Coach Cluster ID: {selected_cluster}")

cluster_coaches = coach_features[coach_features["Cluster"] == selected_cluster].index.tolist()

st.write("Similar Coaches:")
st.write(cluster_coaches)

st.write("Selected coach:", selected_coach)
st.write("Coach feature index sample:", coach_features.index.tolist()[:10])

distance_matrix = euclidean_distances(X_scaled)

distance_df = pd.DataFrame(
    distance_matrix,
    index=coach_features.index,
    columns=coach_features.index
)

st.subheader("Most Similar Coaches")

if selected_coach in distance_df.index:

    similarities = (
        distance_df[selected_coach]
        .sort_values()
        .drop(selected_coach)
        .head(5)
    )

    similar_df = pd.DataFrame({
        "Coach": similarities.index,
        "Similarity Score": similarities.values.round(2)
    })

    st.dataframe(similar_df)

else:
    st.error("Selected coach not found in similarity engine")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coach_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
coach_features["Cluster"] = kmeans.fit_predict(X_scaled)

dna_df = coach_features.reset_index()

st.subheader("Coach DNA Map")

fig = px.scatter(
    dna_df,
    x="xGF_60",
    y="xGA_60",
    color="Cluster",
    hover_name="Head Coach",
    size="xG_pct",
)

# Highlight selected coach
selected_row = dna_df[dna_df["Head Coach"] == selected_coach]

if not selected_row.empty:
    fig.add_scatter(
        x=selected_row["xGF_60"],
        y=selected_row["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=["YOU"],
        textposition="top center",
        name="Selected Coach"
    )

st.plotly_chart(fig, use_container_width=True)

st.subheader("Replacement Recommendations")

if selected_coach in distance_df.index:

    replacements = (
        distance_df[selected_coach]
        .sort_values()
        .drop(selected_coach)
        .head(3)
    )

    replacement_df = pd.DataFrame({
        "Replacement Coach": replacements.index,
        "Fit Score": (100 - replacements.values * 10).round(1)
    })

    st.dataframe(replacement_df)

top_replacement = replacement_df.iloc[0]["Replacement Coach"]

st.success(
    f"{top_replacement} profiles as the strongest stylistic replacement for {selected_coach}."
)

st.subheader("Coach Intelligence Summary")

coach_profile = coach_features.loc[selected_coach]

offense = round(coach_profile["xGF_60"], 2)
defense = round(coach_profile["xGA_60"], 2)
xg = round(coach_profile["xG_pct"], 2)
pdo = round(coach_profile["PDO"], 3)

# --- DISPLAY METRICS ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("xGF/60", offense)
col2.metric("xGA/60", defense)
col3.metric("xG%", xg)
col4.metric("PDO", pdo)

st.divider()

# --- NARRATIVE ENGINE ---
narrative = []

# Offensive identity
if offense > coach_features["xGF_60"].mean():
    narrative.append(
        "This coach emphasizes offensive pressure and chance generation."
    )
else:
    narrative.append(
        "This coach prefers a more controlled offensive structure."
    )

# Defensive identity
if defense < coach_features["xGA_60"].mean():
    narrative.append(
        "Defensively, their teams suppress chances effectively."
    )
else:
    narrative.append(
        "Their defensive structure allows elevated scoring opportunities."
    )

# xG control
if xg > coach_features["xG_pct"].mean():
    narrative.append(
        "Expected-goal control trends above league average."
    )
else:
    narrative.append(
        "Expected-goal control trends below league average."
    )

# PDO sustainability
if pdo > 1.02:
    narrative.append(
        "Performance may be inflated by unusually strong PDO results."
    )
elif pdo < 0.98:
    narrative.append(
        "Results may be partially suppressed by poor PDO variance."
    )

# --- OUTPUT NARRATIVE ---
for sentence in narrative:
    st.write("•", sentence)
