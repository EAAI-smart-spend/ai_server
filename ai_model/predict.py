import torch
from transformers import pipeline
import re
import os

model_path = "./saved_models/fine_tuned_hk_classifier"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model folder {model_path} does not existtttttt!\n"
    )

classifier = None
id2label = {0: "交通", 1: "食飯", 2: "購物", 3: "娛樂", 4: "其他"}

def load_classifier():
    global classifier
    if classifier is None:
        print("Loading model!")
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=False
        )

def extract_total_price(rec_texts):
    text = " ".join(rec_texts)
    patterns = [
        r'(Subtotal|總金額|總計|合計|Total|Amount|PAID)[:：]?\s*\$?\s*(\d+(?:\.\d+)?)',
        r'\$?\s*(\d+(?:\.\d+)?)\s*(Subtotal|總金額|總計|合計|Total|PAID)',
        r'(\d+(?:\.\d+)?)\s*元',
    ]
    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                price_str = [m for m in match if m and m.replace('.', '').isdigit()][0]
            else:
                price_str = match
            candidates.append(float(price_str))
    
    if candidates:
        return max(candidates)
    return None

def getLResult(text: str):
    load_classifier()
    result = classifier(text)[0]
    label = result["label"]
    if label.startswith("LABEL_"):
        label_id = int(label.replace("LABEL_", ""))
        category = id2label.get(label_id, "其他")
    else:
        category = label
    confidence = result["score"]
    total_price = extract_total_price(text.split('\n'))
    return {
        "category": category,
        "confidence": confidence,
        "amount": total_price
    }