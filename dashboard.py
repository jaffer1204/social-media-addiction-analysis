import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Social Media Addiction Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load the social media addiction dataset."""
    return pd.read_csv("./dataset/Students Social Media Addiction.csv")


df = load_data()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("Dashboard Controls")

theme_mode = st.sidebar.radio(
    "Theme",
    ["Light", "Dark"],
    horizontal=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

gender_options = sorted(df["Gender"].dropna().unique())
platform_options = sorted(df["Most_Used_Platform"].dropna().unique())

selected_gender = st.sidebar.multiselect(
    "Gender",
    options=gender_options,
    default=gender_options,
)

selected_age = st.sidebar.slider(
    "Age Range",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
)

selected_platform = st.sidebar.multiselect(
    "Most Used Platform",
    options=platform_options,
    default=platform_options,
)

selected_addiction_score = st.sidebar.slider(
    "Addiction Score Range",
    min_value=int(df["Addicted_Score"].min()),
    max_value=int(df["Addicted_Score"].max()),
    value=(int(df["Addicted_Score"].min()), int(df["Addicted_Score"].max())),
)


# ============================================================
# THEME STYLING
# ============================================================
if theme_mode == "Dark":
    app_bg = "#0f172a"
    card_bg = "rgba(30, 41, 59, 0.78)"
    text_color = "#f8fafc"
    muted_text = "#cbd5e1"
    border_color = "rgba(148, 163, 184, 0.28)"
    hero_gradient = "linear-gradient(135deg, #0f172a 0%, #1e40af 58%, #0f766e 100%)"
    plot_template = "plotly_dark"
else:
    app_bg = "#f5f7fb"
    card_bg = "rgba(255, 255, 255, 0.82)"
    text_color = "#0f172a"
    muted_text = "#64748b"
    border_color = "rgba(226, 232, 240, 0.95)"
    hero_gradient = "linear-gradient(135deg, #102a43 0%, #2563eb 58%, #14b8a6 100%)"
    plot_template = "plotly_white"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {app_bg};
        color: {text_color};
    }}

    .block-container {{
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }}

    .hero {{
        background: {hero_gradient};
        border-radius: 16px;
        padding: 30px 34px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 20px 45px rgba(2, 6, 23, 0.20);
    }}

    .hero-title {{
        font-size: 2.55rem;
        line-height: 1.1;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: 0;
    }}

    .hero-subtitle {{
        font-size: 1rem;
        max-width: 1050px;
        color: #dbeafe;
    }}

    .section-title {{
        color: {text_color};
        font-size: 1.35rem;
        font-weight: 780;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }}

    .kpi-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 19px 20px;
        min-height: 128px;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.10);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
        border-color: #38bdf8;
    }}

    .kpi-label {{
        color: {muted_text};
        font-size: 0.86rem;
        font-weight: 650;
        margin-bottom: 9px;
    }}

    .kpi-value {{
        color: {text_color};
        font-size: 1.72rem;
        line-height: 1.18;
        font-weight: 850;
    }}

    .kpi-note {{
        color: {muted_text};
        font-size: 0.8rem;
        margin-top: 9px;
    }}

    .insight-box {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 13px 15px;
        color: {text_color};
        margin-top: 0.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }}

    .small-muted {{
        color: {muted_text};
        font-size: 0.9rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTER DATA
# ============================================================
filtered_df = df[
    (df["Gender"].isin(selected_gender))
    & (df["Most_Used_Platform"].isin(selected_platform))
    & (df["Age"].between(selected_age[0], selected_age[1]))
    & (
        df["Addicted_Score"].between(
            selected_addiction_score[0],
            selected_addiction_score[1],
        )
    )
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")
st.sidebar.caption(f"Original dataset: {len(df):,} records")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def kpi_card(label, value, note):
    """Create one animated KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text):
    """Display a professional insight below charts."""
    st.markdown(
        f"""
        <div class="insight-box">
            <strong>Insight:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=430):
    """Apply consistent Plotly styling across the dashboard."""
    fig.update_layout(
        template=plot_template,
        height=height,
        margin=dict(l=25, r=25, t=60, b=35),
        font=dict(family="Arial", size=13),
        title_font=dict(size=19),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_color="#ffffff",
            font_size=13,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data
def train_model(data):
    """Train a Random Forest model and return accuracy and feature importance."""
    model_data = data.copy()

    # Student_ID is an identifier, so it is removed before model training.
    if "Student_ID" in model_data.columns:
        model_data = model_data.drop(columns=["Student_ID"])

    # Convert text columns into numeric values.
    label_encoder = LabelEncoder()
    for column in model_data.select_dtypes(include=["object", "string"]).columns:
        model_data[column] = label_encoder.fit_transform(model_data[column])

    X = model_data.drop("Addicted_Score", axis=1)
    y = model_data["Addicted_Score"]

    # Stratified splitting is useful, but it needs at least two samples
    # in every class. If a class has only one record, normal splitting is safer.
    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    prediction_summary = pd.DataFrame(
        {
            "Actual Score": y_test,
            "Predicted Score": predictions,
        }
    )

    return accuracy, importance_df, prediction_summary


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Social Media Addiction Analysis Using Python</div>
        <div class="hero-subtitle">
            A professional interactive analytics dashboard for exploring student
            social media behavior, addiction scores, sleep patterns, mental health,
            relationship status, and machine learning insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust the sidebar filters.")
    st.stop()


# ============================================================
# KPI CARDS
# ============================================================
total_users = len(filtered_df)
avg_addiction = filtered_df["Addicted_Score"].mean()
avg_usage = filtered_df["Avg_Daily_Usage_Hours"].mean()
avg_sleep = filtered_df["Sleep_Hours_Per_Night"].mean()
most_used_platform = filtered_df["Most_Used_Platform"].mode()[0]

st.markdown('<div class="section-title">Executive KPI Overview</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    kpi_card("Total Users", f"{total_users:,}", "Filtered student records")

with kpi2:
    kpi_card("Avg Addiction Score", f"{avg_addiction:.2f}", "Mean addiction level")

with kpi3:
    kpi_card("Avg Daily Usage", f"{avg_usage:.2f} hrs", "Average usage per day")

with kpi4:
    kpi_card("Most Used Platform", most_used_platform, "Leading platform")

with kpi5:
    kpi_card("Avg Sleep Hours", f"{avg_sleep:.2f} hrs", "Average sleep per night")


# ============================================================
# TABS
# ============================================================
overview_tab, charts_tab, ml_tab, data_tab = st.tabs(
    ["Overview", "Interactive Charts", "Machine Learning", "Dataset"]
)


# ============================================================
# OVERVIEW TAB
# ============================================================
with overview_tab:
    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)

    summary_col, chart_col = st.columns([1.1, 1])

    with summary_col:
        numeric_columns = [
            "Age",
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Mental_Health_Score",
            "Conflicts_Over_Social_Media",
            "Addicted_Score",
        ]
        st.dataframe(
            filtered_df[numeric_columns].describe().round(2),
            use_container_width=True,
        )

    with chart_col:
        platform_summary = (
            filtered_df["Most_Used_Platform"].value_counts().reset_index()
        )
        platform_summary.columns = ["Platform", "Users"]

        fig = px.pie(
            platform_summary,
            names="Platform",
            values="Users",
            title="Platform Share",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Users: %{value}<br>Share: %{percent}<extra></extra>",
        )
        fig = style_chart(fig, height=390)
        st.plotly_chart(fig, use_container_width=True)

    insight(
        f"The filtered group has an average daily usage of {avg_usage:.2f} hours "
        f"and an average addiction score of {avg_addiction:.2f}. "
        f"The most common platform is {most_used_platform}."
    )

    st.markdown('<div class="section-title">Quick Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(15), use_container_width=True)


# ============================================================
# INTERACTIVE CHARTS TAB
# ============================================================
with charts_tab:
    st.markdown('<div class="section-title">Usage and Addiction Analysis</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        platform_counts = filtered_df["Most_Used_Platform"].value_counts().reset_index()
        platform_counts.columns = ["Platform", "Users"]

        fig = px.bar(
            platform_counts,
            x="Platform",
            y="Users",
            color="Platform",
            text="Users",
            title="Platform Usage Bar Chart",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Users: %{y}<extra></extra>",
        )
        fig.update_layout(showlegend=False)
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            f"{most_used_platform} has the highest user count in the current filtered view."
        )

    with row1_col2:
        fig = px.histogram(
            filtered_df,
            x="Addicted_Score",
            nbins=10,
            title="Addiction Score Histogram",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(
            xaxis_title="Addiction Score",
            yaxis_title="Number of Users",
        )
        fig.update_traces(
            hovertemplate="Addiction Score: %{x}<br>Users: %{y}<extra></extra>",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            f"The average addiction score is {avg_addiction:.2f}. "
            "Use the sidebar score filter to study low-risk and high-risk groups separately."
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        usage_by_age = (
            filtered_df.groupby("Age", as_index=False)["Avg_Daily_Usage_Hours"].mean()
        )

        fig = px.line(
            usage_by_age,
            x="Age",
            y="Avg_Daily_Usage_Hours",
            markers=True,
            title="Daily Usage Trends by Age",
            color_discrete_sequence=["#14b8a6"],
        )
        fig.update_traces(
            hovertemplate="Age: %{x}<br>Avg Usage: %{y:.2f} hrs<extra></extra>",
        )
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Average Daily Usage Hours",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "This trend chart shows how average daily usage changes across student ages."
        )

    with row2_col2:
        fig = px.scatter(
            filtered_df,
            x="Avg_Daily_Usage_Hours",
            y="Sleep_Hours_Per_Night",
            color="Addicted_Score",
            size="Addicted_Score",
            hover_data=[
                "Gender",
                "Age",
                "Most_Used_Platform",
                "Mental_Health_Score",
            ],
            title="Sleep vs Usage Scatterplot",
            color_continuous_scale="Turbo",
        )
        fig.update_layout(
            xaxis_title="Average Daily Usage Hours",
            yaxis_title="Sleep Hours Per Night",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            f"Average sleep in the filtered data is {avg_sleep:.2f} hours. "
            "Hover over points to compare each student's usage and addiction score."
        )

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        fig = px.scatter(
            filtered_df,
            x="Mental_Health_Score",
            y="Addicted_Score",
            color="Most_Used_Platform",
            size="Avg_Daily_Usage_Hours",
            hover_data=[
                "Gender",
                "Age",
                "Sleep_Hours_Per_Night",
                "Relationship_Status",
            ],
            title="Mental Health vs Addiction Score",
        )
        fig.update_layout(
            xaxis_title="Mental Health Score",
            yaxis_title="Addiction Score",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "This chart helps compare mental health scores with addiction levels "
            "while also showing platform and daily usage behavior."
        )

    with row3_col2:
        gender_summary = (
            filtered_df.groupby("Gender", as_index=False)
            .agg(
                Average_Addiction=("Addicted_Score", "mean"),
                Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
                Users=("Gender", "count"),
            )
        )

        fig = px.bar(
            gender_summary,
            x="Gender",
            y=["Average_Addiction", "Average_Usage"],
            barmode="group",
            title="Gender Comparison Chart",
            hover_data=["Users"],
            color_discrete_sequence=["#2563eb", "#14b8a6"],
        )
        fig.update_layout(
            xaxis_title="Gender",
            yaxis_title="Average Value",
            legend_title="Metric",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "The gender comparison chart displays both average addiction score "
            "and average daily usage for each gender group."
        )

    row4_col1, row4_col2 = st.columns(2)

    with row4_col1:
        numeric_df = filtered_df.select_dtypes(include=["number"])
        correlation = numeric_df.corr().round(2)

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=correlation.values,
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Correlation Heatmap")
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "Positive correlations indicate variables that increase together; "
            "negative correlations indicate inverse relationships."
        )

    with row4_col2:
        relationship_summary = (
            filtered_df.groupby("Relationship_Status", as_index=False)
            .agg(
                Average_Addiction=("Addicted_Score", "mean"),
                Average_Usage=("Avg_Daily_Usage_Hours", "mean"),
                Users=("Relationship_Status", "count"),
            )
            .sort_values("Average_Addiction", ascending=False)
        )

        fig = px.bar(
            relationship_summary,
            x="Relationship_Status",
            y="Average_Addiction",
            color="Average_Usage",
            text="Average_Addiction",
            title="Relationship Status Analysis",
            color_continuous_scale="Viridis",
            hover_data=["Users", "Average_Usage"],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(
            xaxis_title="Relationship Status",
            yaxis_title="Average Addiction Score",
        )
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "Relationship status analysis compares average addiction score "
            "across different student relationship groups."
        )


# ============================================================
# MACHINE LEARNING TAB
# ============================================================
with ml_tab:
    st.markdown('<div class="section-title">Machine Learning Model</div>', unsafe_allow_html=True)

    with st.expander("Model details", expanded=True):
        st.write(
            "A Random Forest Classifier is trained to predict the `Addicted_Score` "
            "using the available student behavior and demographic features. "
            "`Student_ID` is removed because it is only an identifier."
        )

    accuracy, importance_df, prediction_summary = train_model(df)

    ml_col1, ml_col2, ml_col3 = st.columns(3)

    with ml_col1:
        kpi_card("Model Accuracy", f"{accuracy * 100:.2f}%", "Random Forest Classifier")

    with ml_col2:
        kpi_card(
            "Top Feature",
            importance_df.iloc[0]["Feature"],
            "Most important model input",
        )

    with ml_col3:
        exact_predictions = (
            prediction_summary["Actual Score"] == prediction_summary["Predicted Score"]
        ).sum()
        kpi_card(
            "Correct Predictions",
            f"{exact_predictions}/{len(prediction_summary)}",
            "On test dataset",
        )

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    )
    fig = style_chart(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    insight(
        f"The model achieved {accuracy * 100:.2f}% accuracy. "
        f"The strongest feature in the model is `{importance_df.iloc[0]['Feature']}`, "
        "which means it had the highest influence during prediction."
    )

    with st.expander("Prediction sample"):
        st.dataframe(prediction_summary.head(20), use_container_width=True)


# ============================================================
# DATASET TAB
# ============================================================
with data_tab:
    st.markdown('<div class="section-title">Filtered Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_social_media_addiction_data.csv",
        mime="text/csv",
    )

    with st.expander("Dataset information"):
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write("Dataset Shape")
            st.write(f"Rows: {filtered_df.shape[0]}")
            st.write(f"Columns: {filtered_df.shape[1]}")

        with info_col2:
            st.write("Missing Values")
            missing_values = filtered_df.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            st.dataframe(missing_values, use_container_width=True)

    with st.expander("Full summary statistics"):
        st.dataframe(filtered_df.describe(include="all").round(2), use_container_width=True)

    insight(
        "The dataset tab supports data validation, filtered exports, and quick review "
        "of the records used in the dashboard visuals."
    )
