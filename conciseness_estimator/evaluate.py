# evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils.preprocessing import preprocess_dataset
from utils.dataset import PromptDataset
from model import ConcisenessClassifier

X_train, X_test, y_train, y_test, label_encoder = preprocess_dataset()
test_dataset = PromptDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConcisenessClassifier().to(device)
model.load_state_dict(torch.load("conciseness_model.pt"))
model.eval()

y_pred, y_true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(labels)

print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
