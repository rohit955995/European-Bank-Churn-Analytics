import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing function
from data_preprocessing import load_and_preprocess_data


# ==============================
# LOAD PREPROCESSED DATA
# ==============================

df = load_and_preprocess_data()


# ==============================
# BASIC DATASET INFORMATION
# ==============================

print("\n========== DATASET INFO ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())


# ==============================
# TARGET DISTRIBUTION
# ==============================

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Exited"].value_counts())


# ==============================
# OVERALL CHURN RATE
# ==============================

churn_rate = df["Exited"].mean() * 100

print("\n========== OVERALL CHURN RATE ==========")
print(f"Churn Rate: {churn_rate:.2f}%")


# ==============================
# CHURN BY GENDER
# ==============================

print("\n========== CHURN BY GENDER ==========")

churn_gender = df.groupby("Gender")["Exited"].mean() * 100

print(churn_gender)


# ==============================
# CHURN BY GEOGRAPHY
# ==============================

print("\n========== CHURN BY GEOGRAPHY ==========")

churn_geo = df.groupby("Geography")["Exited"].mean() * 100

print(churn_geo)


# ==============================
# CHURN BY AGE GROUP
# ==============================

print("\n========== CHURN BY AGE GROUP ==========")

churn_age = df.groupby("AgeGroup")["Exited"].mean() * 100

print(churn_age)


# ==============================
# CHURN BY CREDIT SCORE GROUP
# ==============================

print("\n========== CHURN BY CREDIT SCORE GROUP ==========")

churn_credit = df.groupby("CreditScoreGroup")["Exited"].mean() * 100

print(churn_credit)


# ==============================
# CHURN BY TENURE GROUP
# ==============================

print("\n========== CHURN BY TENURE GROUP ==========")

churn_tenure = df.groupby("TenureGroup")["Exited"].mean() * 100

print(churn_tenure)


# ==============================
# CHURN BY BALANCE SEGMENT
# ==============================

print("\n========== CHURN BY BALANCE SEGMENT ==========")

churn_balance = df.groupby("BalanceSegment")["Exited"].mean() * 100

print(churn_balance)


# ==============================
# HIGH-VALUE CUSTOMER CHURN
# ==============================

print("\n========== HIGH-VALUE CUSTOMER CHURN ==========")

high_value = df[df["Balance"] > df["Balance"].median()]

high_value_churn = high_value["Exited"].mean() * 100

print(f"High-Value Customer Churn Rate: {high_value_churn:.2f}%")


# ==============================
# CHURNED VS RETAINED
# ==============================

print("\n========== CHURNED VS RETAINED ==========")

print(
    df["Exited"]
    .map({0: "Retained", 1: "Churned"})
    .value_counts()
)


# ==============================
# VISUALIZATIONS
# ==============================

sns.set_theme(style="whitegrid")


# 1. Overall Churn Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Exited"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# 2. Churn by Gender

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_gender.index,
    y=churn_gender.values
)

plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# 3. Churn by Geography

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_geo.index,
    y=churn_geo.values
)

plt.title("Churn Rate by Geography")
plt.xlabel("Geography")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# 4. Churn by Age Group

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_age.index,
    y=churn_age.values
)

plt.title("Churn Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# 5. Churn by Credit Score

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_credit.index,
    y=churn_credit.values
)

plt.title("Churn Rate by Credit Score Group")
plt.xlabel("Credit Score Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# 6. Churn by Tenure

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_tenure.index,
    y=churn_tenure.values
)

plt.title("Churn Rate by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# 7. Churn by Balance Segment

plt.figure(figsize=(7, 5))

sns.barplot(
    x=churn_balance.index,
    y=churn_balance.values
)

plt.title("Churn Rate by Balance Segment")
plt.xlabel("Balance Segment")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


print("\n========== EDA COMPLETED SUCCESSFULLY ==========")