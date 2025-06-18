# Function to check and report duplicate rows in a dataset
def check_duplicates_report(df, dataset_name):
    duplicates_count = df.duplicated().sum()  # Count fully duplicated rows
    total_rows = len(df)

    print("=" * 40)
    print(f"{dataset_name} Duplicate Analysis")
    print("=" * 40)

    if duplicates_count == 0:
        print(f"No duplicates found in {total_rows:,} rows")
    else:
        print(f"{duplicates_count} duplicates found ({duplicates_count/total_rows:.2%})")
        print(f"Total rows affected: {duplicates_count:,}/{total_rows:,}")
