import torch
from transformers import BertTokenizer
from conciseness_estimator.model import ConcisenessClassifier
import torch.nn.functional as F
import pandas as pd

# Load label encoder classes (manually set to match training)
label_encoder_classes = ["cod_strict", "cod_moderate", "cod_soft"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConcisenessClassifier().to(device)
model.load_state_dict(torch.load("conciseness_model_final.pt", map_location=device))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_prompt_strategy(question: str):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        strategy = label_encoder_classes[predicted_class]
    
    return strategy

# Example usage
# if __name__ == "__main__":
#     question = "How does gradient descent optimize a loss function in neural networks?"
#     strategy = predict_prompt_strategy(question)
#     print(f"Recommended prompting strategy: {strategy}")
