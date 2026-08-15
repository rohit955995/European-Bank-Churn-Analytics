import pandas as pd


def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("data/European_Bank.csv")

    # Remove unnecessary columns
    df = df.drop(columns=["Surname"], errors="ignore")

    # Create Age Group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=["<30", "30-45", "46-60", "60+"]
    )

    # Create Credit Score Group
    df["CreditScoreGroup"] = pd.cut(
        df["CreditScore"],
        bins=[0, 580, 670, 850],
        labels=["Low", "Medium", "High"]
    )

    # Create Tenure Group
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[-1, 2, 5, 10],
        labels=["New", "Mid-term", "Long-term"]
    )

    # Create Balance Segment
    df["BalanceSegment"] = pd.cut(
        df["Balance"],
        bins=[-1, 0, 50000, float("inf")],
        labels=["Zero-balance", "Low-balance", "High-balance"]
    )

    return df


if __name__ == "__main__":

    df = load_and_preprocess_data()

    print("\n========== PREPROCESSED DATA ==========")
    print(df.head())

    print("\n========== NEW COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== AGE GROUP ==========")
    print(df["AgeGroup"].value_counts())

    print("\n========== CREDIT SCORE GROUP ==========")
    print(df["CreditScoreGroup"].value_counts())

    print("\n========== TENURE GROUP ==========")
    print(df["TenureGroup"].value_counts())

    print("\n========== BALANCE SEGMENT ==========")
    print(df["BalanceSegment"].value_counts())