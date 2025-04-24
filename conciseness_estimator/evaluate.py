import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

from utils.preprocessing import preprocess_dataset
from utils.dataset import PromptDataset
from model import ConcisenessClassifier
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

X_train, X_test, y_train, y_test, label_encoder = preprocess_dataset()
test_dataset = PromptDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConcisenessClassifier().to(device)
model.load_state_dict(torch.load("conciseness_model_final.pt"))
model.eval()

y_pred, y_true = [], []

print("Testing Started...")

with torch.no_grad():
    for idx, batch in enumerate(test_loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(labels)
        print(f"Batch {idx} processed!")

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Save to JSON
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": conf_matrix
}

with open("model_evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to 'evaluation_metrics.json'")
