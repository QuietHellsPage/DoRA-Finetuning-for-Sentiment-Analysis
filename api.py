import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, DebertaForSequenceClassification
from peft import PeftModel, PeftConfig
import uvicorn

# 1. Defining constants
MODEL_PATH = "dora_model/checkpoint-3938"
BASE_MODEL = "deepvk/deberta-v1-base"
DEVICE = "cpu"


# 2. Schema for the input text
class TextInput(BaseModel):
    text: str


# 3. Loading fine-tuned model
peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_model = DebertaForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 4. Creating app and checkpoint
app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment Analysis for the russian clothes reviews",
)


@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    try:
        inputs = tokenizer(
            input_data.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]

        label_names = ["neautral", "negative", "positive"]
        sentiment = label_names[pred]

        return {
            "sentiment": sentiment,
            "confidence": float(max(probs)),
            "probabilities": {label: prob for label, prob in zip(label_names, probs)},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 5. Running checkpoint via uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
