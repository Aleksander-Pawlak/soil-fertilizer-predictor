# Utility to print unique value counts and distribution of categorical features
def print_unique_values(df, categorical_columns, dataset_name="Dataset"):
    print("\n" + "=" * 50)
    print(f"Unique values in {dataset_name} categorical features")
    print("=" * 50)
    for col in categorical_columns:
        unique_values = sorted(df[col].unique())  # List of distinct values
        value_counts = df[col].value_counts()   # Frequency of each category
        top_value = value_counts.index[0]       # Most frequent category
        top_freq = value_counts.iloc[0]         # Frequency of that category
        print(f"{col} - Number of unique values: {len(unique_values)}")
        print(f"Unique values: {unique_values}")
        print(f"Top value: '{top_value}' (Frequency: {top_freq})\n")

