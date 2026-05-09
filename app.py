import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SHEET_ID = "1JPWoFRyeEEjD-0FFkZP7-DF2aSbKl3oUi8e7S9yF_ns"

@st.cache_data
def load_data(sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url)

stats = load_data("RawStats")
coaches = load_data("Coach_Registry")

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

coach_features = stats.groupby("Coach")[[
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

selected_cluster = coach_features.loc[selected_coach, "Cluster"]

st.write(f"Coach Cluster ID: {selected_cluster}")

cluster_coaches = coach_features[coach_features["Cluster"] == selected_cluster].index.tolist()

st.write("Similar Coaches:")
st.write(cluster_coaches)
