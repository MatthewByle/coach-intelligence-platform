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
# CLEAN
# =========================================================
stats.columns = stats.columns.astype(str).str.strip()
coaches.columns = coaches.columns.astype(str).str.strip()

stats = stats.loc[:, ~stats.columns.str.contains("^Unnamed")]

if "Coach" not in stats.columns:
    stats = stats.rename(columns={stats.columns[0]: "Coach"})

stats["Coach"] = stats["Coach"].astype(str).str.strip()
coaches["Head Coach"] = coaches["Head Coach"].astype(str).str.strip()

for col in ["xGF_60", "xGA_60", "xG_pct", "PDO"]:
    if col in stats.columns:
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

stats["Date"] = pd.to_datetime(stats.get("Date"), errors="coerce")

# =========================================================
# SIDEBAR
# =========================================================
coach_list = sorted(coaches["Head Coach"].dropna().unique())
selected_sidebar = st.sidebar.selectbox("Select Coach", coach_list)

if st.session_state.active_coach is None:
    st.session_state.active_coach = selected_sidebar

if st.sidebar.button("Reset Focus"):
    st.session_state.active_coach = selected_sidebar

active_coach = st.session_state.active_coach

# =========================================================
# LOOKUP
# =========================================================
coach_row = coaches[coaches["Head Coach"] == active_coach].iloc[0]
team = coach_row["Team Name"]

team_stats = stats[stats["Team"] == team]

# =========================================================
# MODEL (DNA + ROLE CLUSTERS)
# =========================================================
coach_features = stats.groupby("Coach")[[
    "xGF_60", "xGA_60", "xG_pct", "PDO"
]].mean().dropna()

X = StandardScaler().fit_transform(coach_features)

coach_features["Cluster"] = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
).fit_predict(X)

dna_df = coach_features.reset_index()

# =========================================================
# ROLE DEFINITIONS (NEW CORE)
# =========================================================
def get_role(cluster):
    return {
        0: "Balanced System Coach",
        1: "Offensive Pressure Coach",
        2: "Defensive Structure Coach",
        3: "High-Variance Transition Coach"
    }.get(cluster, "Unknown")

dna_df["Role"] = dna_df["Cluster"].apply(get_role)

# =========================================================
# ROLE-BASED GRADING ENGINE (NEW CORE)
# =========================================================
def role_grade(df, coach_name, metric, higher_is_better=True):
    group = df.groupby("Role")[metric].apply(list)

    coach_role = dna_df[dna_df["Coach"] == coach_name]["Role"].values[0]

    role_values = group[coach_role]
    coach_value = df[df["Coach"] == coach_name][metric].mean()

    if higher_is_better:
        pct = (pd.Series(role_values) <= coach_value).mean()
    else:
        pct = (pd.Series(role_values) >= coach_value).mean()

    if pct >= 0.85:
        return "A"
    elif pct >= 0.70:
        return "B"
    elif pct >= 0.55:
        return "C"
    elif pct >= 0.40:
        return "D"
    else:
        return "F"

coach_role = dna_df[dna_df["Coach"] == active_coach]["Role"].values[0]

off_grade = role_grade(stats, active_coach, "xGF_60", True)
def_grade = role_grade(stats, active_coach, "xGA_60", False)
sys_grade = role_grade(stats, active_coach, "xG_pct", True)

# overall role-based grade
grade_map = {"A":5,"B":4,"C":3,"D":2,"F":1}
overall_score = (grade_map[off_grade] + grade_map[def_grade] + grade_map[sys_grade]) / 3

if overall_score >= 4.5:
    overall_grade = "A"
elif overall_score >= 3.5:
    overall_grade = "B"
elif overall_score >= 2.5:
    overall_grade = "C"
elif overall_score >= 1.5:
    overall_grade = "D"
else:
    overall_grade = "F"

# =========================================================
# HEADER
# =========================================================
st.subheader(f"Coach: {active_coach}")
st.write(f"Team: **{team}**")
st.write(f"Role: **{coach_role}**")

# =========================================================
# SCORECARD (ROLE BASED)
# =========================================================
st.subheader("Coach Scorecard (Role-Based)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offense Grade", off_grade)
c2.metric("Defense Grade", def_grade)
c3.metric("System Grade", sys_grade)
c4.metric("Overall Grade", overall_grade)

# =========================================================
# TEAM STATS
# =========================================================
st.subheader("Team Stats")

profile = team_stats[["xGF_60","xGA_60","xG_pct","PDO"]].mean(numeric_only=True)

cols = st.columns(4)
cols[0].metric("xGF/60", round(profile["xGF_60"],2))
cols[1].metric("xGA/60", round(profile["xGA_60"],2))
cols[2].metric("xG%", round(profile["xG_pct"],2))
cols[3].metric("PDO", round(profile["PDO"],3))

# =========================================================
# DNA MAP
# =========================================================
st.subheader("Coach DNA Map")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dna_df["xGF_60"],
    y=dna_df["xGA_60"],
    mode="markers",
    text=dna_df["Coach"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    marker=dict(size=10, opacity=0.4)
))

active_point = dna_df[dna_df["Coach"] == active_coach]

if not active_point.empty:
    fig.add_trace(go.Scatter(
        x=active_point["xGF_60"],
        y=active_point["xGA_60"],
        mode="markers+text",
        marker=dict(size=18, color="red"),
        text=[active_coach],
        textposition="top center"
    ))

fig.update_layout(hovermode="closest", showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SIMILAR COACHES
# =========================================================
st.subheader("Most Similar Coaches")

distance_df = pd.DataFrame(
    euclidean_distances(X),
    index=coach_features.index,
    columns=coach_features.index
)

if active_coach in distance_df.index:
    sims = distance_df[active_coach].sort_values().drop(active_coach).head(5)

    st.dataframe(pd.DataFrame({
        "Coach": sims.index,
        "Score": sims.values.round(2)
    }))

# =========================================================
# REPLACEMENTS
# =========================================================
st.subheader("Replacement Candidates")

replacement = dna_df[dna_df["Role"] == coach_role].head(5)

st.dataframe(replacement[["Coach"]])

# =========================================================
# AI SUMMARY
# =========================================================
st.subheader("Coaching Summary")

st.write(f"""
- Role: **{coach_role}**
- Offensive Grade: **{off_grade}**
- Defensive Grade: **{def_grade}**
- System Grade: **{sys_grade}**
- Overall Grade: **{overall_grade}**
""")
