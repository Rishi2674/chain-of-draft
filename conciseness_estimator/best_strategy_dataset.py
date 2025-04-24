import pandas as pd

def deduplicate_by_best_strategy(path="conciseness_estimator/dataset.csv", save_path="conciseness_estimator/best_strategy_dataset.csv"):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    
    valid_templates = {"cod_soft", "cod_moderate", "cod_strict"}
    df = df[df["prompt_template"].isin(valid_templates)]

    # Filter only correct answers
    df = df[df["is_correct"] == True]

    # Keep the row with the lowest token_count for each question
    df_best = df.loc[df.groupby("question")["token_count"].idxmin()]

    # Save or return
    df_best.to_csv(save_path, index=False)
    return df_best

# Example usage
deduplicated_df = deduplicate_by_best_strategy()
