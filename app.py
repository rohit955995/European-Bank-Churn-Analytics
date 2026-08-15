import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_and_preprocess_data


# PAGE CONFIG

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)


# LOAD + PREPROCESS DATA

df = load_and_preprocess_data()


# TITLE

st.title("🏦 European Bank Customer Churn Analytics")

st.write(
    "Interactive dashboard for analysing customer segmentation "
    "and churn patterns in European banking."
)

# SIDEBAR FILTERS

st.sidebar.header("🎛️ Filters")

geography = st.sidebar.multiselect(
    "Geography",
    options=sorted(df["Geography"].unique()),
    default=sorted(df["Geography"].unique())
)

gender = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

age_group = st.sidebar.multiselect(
    "Age Group",
    options=sorted(df["AgeGroup"].unique()),
    default=sorted(df["AgeGroup"].unique())
)

balance_segment = st.sidebar.multiselect(
    "Balance Segment",
    options=sorted(df["BalanceSegment"].unique()),
    default=sorted(df["BalanceSegment"].unique())
)


# APPLY FILTERS

filtered_df = df[
    (df["Geography"].isin(geography)) &
    (df["Gender"].isin(gender)) &
    (df["AgeGroup"].isin(age_group)) &
    (df["BalanceSegment"].isin(balance_segment))
].copy()


# KPI CALCULATIONS

total_customers = len(filtered_df)

churned_customers = filtered_df["Exited"].sum()

if total_customers > 0:
    churn_rate = (churned_customers / total_customers) * 100
else:
    churn_rate = 0

if len(filtered_df[filtered_df["BalanceSegment"] == "High-balance"]) > 0:
    high_value_df = filtered_df[
        filtered_df["BalanceSegment"] == "High-balance"
    ]

    high_value_churn = (
        high_value_df["Exited"].mean() * 100
    )
else:
    high_value_churn = 0


# KPI SECTION

st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{int(churned_customers):,}"
)

col3.metric(
    "Overall Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "High-Value Churn",
    f"{high_value_churn:.2f}%"
)

st.divider()


# CHURN DISTRIBUTION

st.header("📌 Churn Distribution")

col1, col2 = st.columns(2)


# DONUT CHART

with col1:

    churn_data = filtered_df["Exited"].value_counts().reset_index()

    churn_data.columns = ["Exited", "Count"]

    churn_data["Status"] = churn_data["Exited"].map({
        0: "Retained",
        1: "Churned"
    })

    fig = px.pie(
        churn_data,
        values="Count",
        names="Status",
        hole=0.5,
        title="Customer Churn Distribution"
    )

    fig.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# GEOGRAPHY CHURN

with col2:

    geo_churn = (
        filtered_df
        .groupby("Geography")["Exited"]
        .mean()
        .reset_index()
    )

    geo_churn["Churn Rate"] = geo_churn["Exited"] * 100

    fig = px.bar(
        geo_churn,
        x="Geography",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Geography"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_yaxes(
        title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# CUSTOMER SEGMENTATION
# ========================================================
st.header("👥 Customer Segmentation")

col1, col2 = st.columns(2)


# AGE GROUP

with col1:

    age_churn = (
        filtered_df
        .groupby("AgeGroup")["Exited"]
        .mean()
        .reset_index()
    )

    age_churn["Churn Rate"] = age_churn["Exited"] * 100

    fig = px.bar(
        age_churn,
        x="AgeGroup",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Age Group"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# GENDER

with col2:

    gender_churn = (
        filtered_df
        .groupby("Gender")["Exited"]
        .mean()
        .reset_index()
    )

    gender_churn["Churn Rate"] = gender_churn["Exited"] * 100

    fig = px.bar(
        gender_churn,
        x="Gender",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Gender"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# FINANCIAL ANALYSIS

st.header("💰 Financial & Product Analysis")

col1, col2 = st.columns(2)


# BALANCE SEGMENT

with col1:

    balance_churn = (
        filtered_df
        .groupby("BalanceSegment")["Exited"]
        .mean()
        .reset_index()
    )

    balance_churn["Churn Rate"] = (
        balance_churn["Exited"] * 100
    )

    fig = px.bar(
        balance_churn,
        x="BalanceSegment",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Balance Segment"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# NUMBER OF PRODUCTS

with col2:

    product_churn = (
        filtered_df
        .groupby("NumOfProducts")["Exited"]
        .mean()
        .reset_index()
    )

    product_churn["Churn Rate"] = (
        product_churn["Exited"] * 100
    )

    fig = px.bar(
        product_churn,
        x="NumOfProducts",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Number of Products"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# CUSTOMER ACTIVITY

st.header("👤 Customer Activity")

col1, col2 = st.columns(2)


# ACTIVE MEMBER

with col1:

    active_churn = (
        filtered_df
        .groupby("IsActiveMember")["Exited"]
        .mean()
        .reset_index()
    )

    active_churn["Member Status"] = active_churn[
        "IsActiveMember"
    ].map({
        0: "Inactive",
        1: "Active"
    })

    active_churn["Churn Rate"] = (
        active_churn["Exited"] * 100
    )

    fig = px.bar(
        active_churn,
        x="Member Status",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate: Active vs Inactive Members"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# CREDIT SCORE

with col2:

    credit_churn = (
        filtered_df
        .groupby("CreditScoreGroup")["Exited"]
        .mean()
        .reset_index()
    )

    credit_churn["Churn Rate"] = (
        credit_churn["Exited"] * 100
    )

    fig = px.bar(
        credit_churn,
        x="CreditScoreGroup",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Credit Score"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# BUSINESS INSIGHTS

st.header("💡 Business Insights")

if total_customers > 0:

    highest_geo = (
        filtered_df
        .groupby("Geography")["Exited"]
        .mean()
        .idxmax()
    )

    highest_age = (
        filtered_df
        .groupby("AgeGroup")["Exited"]
        .mean()
        .idxmax()
    )

    highest_gender = (
        filtered_df
        .groupby("Gender")["Exited"]
        .mean()
        .idxmax()
    )

    st.info(
        f"""
        📍 **Highest geographical churn:** {highest_geo}

        👥 **Highest-risk age group:** {highest_age}

        🚻 **Higher churn gender:** {highest_gender}

        📊 **Overall churn rate:** {churn_rate:.2f}%

        💰 **High-value customer churn:** {high_value_churn:.2f}%
        """
    )


# DATA PREVIEW

with st.expander("🔍 View Filtered Customer Data"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# FOOTER

st.divider()

st.caption(
    "European Bank Customer Churn Analytics | "
    "Machine Learning Project"
)

# CUSTOMER CHURN PREDICTION


import joblib
import pandas as pd
import streamlit as st

st.header("🤖 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to leave the bank."
)

# Load trained model
model = joblib.load("model.joblib")


# Customer Inputs


col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=250000.0,
        value=50000.0
    )



# Prediction


if st.button("🔮 Predict Churn", type="primary"):

    # Convert Yes/No to 1/0
    has_cr_card_value = 1 if has_cr_card == "Yes" else 0
    is_active_value = 1 if is_active == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card_value],
        "IsActiveMember": [is_active_value],
        "EstimatedSalary": [estimated_salary]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()


    # Result


    if prediction == 1:

        st.error("⚠️ HIGH CHURN RISK")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "This customer is likely to leave the bank. "
            "Consider offering retention benefits or personalized services."
        )

    else:

        st.success("✅ LOW CHURN RISK")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "This customer is likely to remain with the bank."
        )

# CUSTOMER SEGMENTATION
# =========================================================

st.divider()

st.header("👥 Customer Segmentation")

st.write(
    "Customers are divided into different segments based on "
    "balance, activity and product usage."
)


# SEGMENT 1: BALANCE


st.subheader("💰 Balance-based Segmentation")

balance_summary = (
    filtered_df
    .groupby("BalanceSegment", observed=True)
    .agg(
        Customers=("CustomerId", "count"),
        Churned=("Exited", "sum"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

balance_summary["Churn_Rate"] = (
    balance_summary["Churn_Rate"] * 100
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        balance_summary,
        x="BalanceSegment",
        y="Customers",
        text="Customers",
        title="Customers by Balance Segment"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.bar(
        balance_summary,
        x="BalanceSegment",
        y="Churn_Rate",
        text="Churn_Rate",
        title="Churn Rate by Balance Segment"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



# SEGMENT 2: CUSTOMER ACTIVITY
# ---------------------------------------------------------

st.subheader("👤 Activity-based Segmentation")

activity_summary = (
    filtered_df
    .groupby("IsActiveMember")
    .agg(
        Customers=("CustomerId", "count"),
        Churned=("Exited", "sum"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

activity_summary["Customer Type"] = activity_summary[
    "IsActiveMember"
].map({
    0: "Inactive",
    1: "Active"
})

activity_summary["Churn_Rate"] = (
    activity_summary["Churn_Rate"] * 100
)

fig = px.bar(
    activity_summary,
    x="Customer Type",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate: Active vs Inactive Customers"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)



# SEGMENT 3: PRODUCT USAGE


st.subheader("📦 Product Usage Segmentation")

product_summary = (
    filtered_df
    .groupby("NumOfProducts")
    .agg(
        Customers=("CustomerId", "count"),
        Churned=("Exited", "sum"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

product_summary["Churn_Rate"] = (
    product_summary["Churn_Rate"] * 100
)

fig = px.bar(
    product_summary,
    x="NumOfProducts",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate by Number of Products"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)



# SEGMENT 4: HIGH-VALUE CUSTOMERS


st.subheader("💎 High-Value Customer Segment")

if len(filtered_df) > 0:

    median_balance = filtered_df["Balance"].median()

    high_value_customers = filtered_df[
        filtered_df["Balance"] > median_balance
    ]

    high_value_total = len(high_value_customers)

    high_value_churned = high_value_customers["Exited"].sum()

    if high_value_total > 0:
        high_value_rate = (
            high_value_churned /
            high_value_total
        ) * 100
    else:
        high_value_rate = 0

else:

    high_value_total = 0
    high_value_churned = 0
    high_value_rate = 0


col1, col2, col3 = st.columns(3)

col1.metric(
    "High-Value Customers",
    f"{high_value_total:,}"
)

col2.metric(
    "High-Value Churned",
    f"{int(high_value_churned):,}"
)

col3.metric(
    "High-Value Churn Rate",
    f"{high_value_rate:.2f}%"
)



# BUSINESS RECOMMENDATIONS


st.divider()

st.header("💡 Business Recommendations")

st.markdown("""
### 🎯 Recommended Actions

**1. High-risk customers**
- Identify customers with high predicted churn probability.
- Provide personalized retention offers.

**2. Inactive customers**
- Send targeted engagement campaigns.
- Encourage customers to use banking products.

**3. High-value customers**
- Provide premium services and personalized support.
- Monitor their churn probability regularly.

**4. Product usage**
- Analyse customers using very few or unusually many products.
- Recommend suitable banking products.

**5. Customer segmentation**
- Use age, geography, balance and activity to create targeted campaigns.
""")

st.success(
    "✅ Customer segmentation and business recommendations completed."
)
