import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.preprocessing import preprocess_dataset
from utils.dataset import PromptDataset
from model import ConcisenessClassifier
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load preprocessed dataset
X_train,X_test, y_train, y_test, label_encoder = preprocess_dataset()

# Initialize the dataset and dataloader
train_dataset = PromptDataset(X_train, y_train, tokenizer)  # Only texts and labels now
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConcisenessClassifier().to(device)  # No extra_feat_dim needed

optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

print("Training started...")
for epoch in range(5):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed.")

# Save the trained model
torch.save(model.state_dict(), "conciseness_model_final.pt")
print("Model saved as conciseness_model_final.pt.")