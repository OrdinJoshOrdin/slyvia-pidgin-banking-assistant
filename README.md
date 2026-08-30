# Naija Pidgin Banking Assistant (Slyvia)

A chat interface backed by a fine-tuned BERT (`bert-base-uncased`) intent classifier for Nigerian Pidgin banking customer-service conversations.

## What it does

Understands customer messages written in Nigerian Pidgin and routes them to the right banking support flow. Recognized intents:

- Account Opening
- ATM Issue
- Balance Inquiry
- Card Issue
- Failed Transfer
- Fraud Report
- Loan Request
- Mobile App Issue
- Out of Scope (anything unrelated to banking gets a clarifying reply instead of a wrong guess)

## Files

- `slyvia_app.py` — the Streamlit chat app
- `saved_models/slyvia_pidgin_bert/` — the trained model, tokenizer, label encoder, and response map
- `train_and_save_slyvia_model.py` — trains the model from scratch and saves it into `saved_models/`
- `SLYVIA.ipynb` — the full ML pipeline notebook (data cleaning, training, evaluation), for reference
- `SLYVIA DATASET2.csv` — the training dataset
- `requirements.txt` — Python dependencies

## Running locally

```bash
pip install -r requirements.txt
streamlit run slyvia_app.py
```

## Evaluation

Trained on 7,650 conversation turns (9 intents), evaluated on a genuinely unseen 81-example test set (not memorized templates) — see `saved_models/slyvia_pidgin_bert/metrics.json` for full precision/recall/F1 per intent.
