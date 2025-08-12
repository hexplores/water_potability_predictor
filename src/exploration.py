import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def save_plot(fig, filename, folder="plots"):
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved plot to {filepath}")

def show_basic_info(df):
    print("🔹 Column Names:")
    print(df.columns.tolist())
    print("\n🔹 First 5 Rows:")
    print(df.head())
    print("\n🔹 Data Types & Non-Null Info:")
    print(df.info())
    print("\n🔹 Missing Values (Count):")
    print(df.isna().sum())
    print("\n🔹 Missing Values (%):")
    print((df.isna().sum() / len(df)) * 100)

    duplicates = df[df.duplicated()]
    print(f"\n🔹 Duplicate Rows: {len(duplicates)}")
    if not duplicates.empty:
        print(duplicates)

def show_boxplots(df, columns):
    for col in columns:
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.boxplot(x=col, data=df, color="skyblue", flierprops=dict(markersize=6, markerfacecolor="black"), ax=ax)
        ax.set_xlabel(col)
        ax.set_title(f"Distribution of {col} Values")
        plt.tight_layout()
        save_plot(fig, f"boxplot_{col}.png")

def fill_missing_with_median(df, columns):
    for col in columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

def plot_potability_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(x="Potability", data=df, ax=ax)
    ax.set_title("Potability Distribution")
    ax.set_xlabel("Potability")
    ax.set_ylabel("Count")
    plt.tight_layout()
    save_plot(fig, "potability_distribution.png")

def plot_correlation_heatmap(df):
    corr = df.corr()
    print("\n🔹 Correlation Matrix:")
    print(corr)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    plt.tight_layout()
    save_plot(fig, "correlation_heatmap.png")

def explore(df):
    """
    Perform basic EDA on a DataFrame.
    """
    show_basic_info(df)

    # Boxplot outlier checks
    outlier_columns = ["ph", "Sulfate", "Trihalomethanes"]
    show_boxplots(df, outlier_columns)

    # Handle missing data
    fill_missing_with_median(df, outlier_columns)

    # Potability distribution
    plot_potability_distribution(df)

    # Correlation heatmap
    plot_correlation_heatmap(df)

    return df
