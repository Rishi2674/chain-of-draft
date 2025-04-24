# utils/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(path="conciseness_estimator/best_strategy_dataset.csv"):
    df = pd.read_csv(path, encoding="ISO-8859-1")

    # Filter valid prompt_templates
    

    # Map templates to labels
    def map_template_to_label(template):
        if "strict" in template:
            return "3_words"
        elif "moderate" in template:
            return "5_words"
        elif "soft" in template:
            return "7_words"
        else:
            return "unknown"

    df["label"] = df["prompt_template"].apply(map_template_to_label)

    # Encode the label
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    # Prepare features and labels
    texts = df["question"].tolist()
    labels = df["label_encoded"].tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoder
