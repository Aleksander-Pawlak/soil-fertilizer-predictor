# Utility to print missing value statistics for a dataset
def missing_values_report(df, dataset_name):
    missing_count = df.isnull().sum().sum()  # Total number of missing entries
    rows = len(df)

    print("=" * 40)
    print(f"{dataset_name} Missing Value Analysis")
    print("=" * 40)

    if missing_count == 0:
        print(f"No missing values detected in {rows:,} rows")
    else:
        print(f"{missing_count} missing values found in {rows:,} rows")

