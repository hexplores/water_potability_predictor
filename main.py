import pandas as pd
from src.exploration import explore
from model import train_and_compare_models
def main():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Water_Quality_prediction\data\water_potability.csv")
    cleaned_df = explore(df)
    results= train_and_compare_models(cleaned_df)
if __name__ == "__main__":
    main()
