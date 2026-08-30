"""
Train the Nigerian Pidgin Banking Intent Classifier (BERT) and save
everything the Streamlit app needs:

  saved_models/slyvia_pidgin_bert/
      model/                   <- BertForSequenceClassification.save_pretrained()
      tokenizer/               <- BertTokenizer.save_pretrained()
      label_encoder.pkl
      response_map.pkl
      config.json              <- max_length, model_name, class names
      metrics.json             <- accuracy/precision/recall/f1 + confusion matrix
                                    + classification report (computed on the
                                    72-example unseen test set)
      training_history.json    <- per-epoch train/val loss & accuracy
"""

import os
import re
import json
import pickle
import time

import pandas as pd
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'saved_models', 'slyvia_pidgin_bert')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60, flush=True)
print("STAGE 1-2: Load and clean data", flush=True)
print("=" * 60, flush=True)

raw_df = pd.read_csv(os.path.join(BASE_DIR, 'SLYVIA DATASET2.csv'), encoding='latin-1')
conversation_df = raw_df.rename(columns={
    'turn': 'turn_number',
    'user_text': 'user_message',
    'suggested_response': 'bot_response'
})


def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


conversation_df['user_message'] = conversation_df['user_message'].apply(clean_text)
conversation_df = conversation_df.dropna(subset=['user_message', 'intent_label', 'bot_response'])
conversation_df = conversation_df.reset_index(drop=True)


def normalize_pidgin_text(text):
    text = str(text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.replace(' dnt ', ' dont ')
    text = text.replace(' wnt ', ' wont ')
    text = text.replace(' cnt ', ' cant ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


conversation_df['user_message'] = conversation_df['user_message'].apply(normalize_pidgin_text)
conversation_df['user_message'] = conversation_df['user_message'].str.lower()

print("Rows after cleaning:", len(conversation_df), flush=True)

print("=" * 60, flush=True)
print("STAGE 3: Build context column", flush=True)
print("=" * 60, flush=True)

conversation_df = conversation_df.sort_values(by=['conversation_id', 'turn_number']).reset_index(drop=True)
conversation_df['context'] = ''
for conv_id, group in conversation_df.groupby('conversation_id'):
    group_sorted = group.sort_values('turn_number')
    indices = group_sorted.index.tolist()
    messages = group_sorted['user_message'].tolist()
    for i, idx in enumerate(indices):
        conversation_df.at[idx, 'context'] = '' if i == 0 else messages[i - 1]

print("Context column built.", flush=True)

print("=" * 60, flush=True)
print("Loading BertTokenizer...", flush=True)
print("=" * 60, flush=True)
from transformers import BertTokenizer
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 64
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_conversation_turn(context, message, max_length=MAX_LENGTH):
    if context.strip() == '':
        encoding = tokenizer(message, padding='max_length', truncation=True,
                              max_length=max_length, return_attention_mask=True)
    else:
        encoding = tokenizer(context, message, padding='max_length', truncation=True,
                              max_length=max_length, return_attention_mask=True)
    return encoding['input_ids'], encoding['attention_mask']


print("=" * 60, flush=True)
print("STAGE 4: Label encoding + tokenization", flush=True)
print("=" * 60, flush=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
conversation_df['encoded_label'] = label_encoder.fit_transform(conversation_df['intent_label'])
print("Classes:", list(label_encoder.classes_), flush=True)

all_input_ids, all_attention_masks = [], []
for row_idx, row in conversation_df.iterrows():
    iid, am = tokenize_conversation_turn(row['context'], row['user_message'])
    all_input_ids.append(iid)
    all_attention_masks.append(am)
    if (row_idx + 1) % 2000 == 0:
        print(f"  Tokenized {row_idx + 1}/{len(conversation_df)}", flush=True)

input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
attention_masks_tensor = torch.tensor(all_attention_masks, dtype=torch.long)
labels_tensor = torch.tensor(conversation_df['encoded_label'].tolist(), dtype=torch.long)

print("=" * 60, flush=True)
print("STAGE 4: Train/Val split (85/15)", flush=True)
print("=" * 60, flush=True)

from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(
    range(len(input_ids_tensor)), test_size=0.15, random_state=42,
    stratify=labels_tensor.numpy())

train_input_ids = input_ids_tensor[train_indices]
train_attention_masks = attention_masks_tensor[train_indices]
train_labels = labels_tensor[train_indices]

val_input_ids = input_ids_tensor[val_indices]
val_attention_masks = attention_masks_tensor[val_indices]
val_labels = labels_tensor[val_indices]

print(f"Train: {len(train_indices)}  Val: {len(val_indices)}", flush=True)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("=" * 60, flush=True)
print("STAGE 5: BERT model init", flush=True)
print("=" * 60, flush=True)

from transformers import BertForSequenceClassification
num_intent_labels = len(label_encoder.classes_)
bert_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_intent_labels,
    output_attentions=False, output_hidden_states=False
)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
bert_model = bert_model.to(device)
print("Device:", device, flush=True)

print("=" * 60, flush=True)
print("STAGE 6: Training (4 epochs)", flush=True)
print("=" * 60, flush=True)

from torch.optim import AdamW
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
NUM_EPOCHS = 4

training_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': []}

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    bert_model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = bert_model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    bert_model.eval()
    total_val_loss, total_correct, total_val_samples = 0, 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = bert_model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            total_val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += (predictions == b_labels).sum().item()
            total_val_samples += b_labels.size(0)

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = total_correct / total_val_samples * 100
    epoch_time = time.time() - epoch_start

    training_history['epoch'].append(epoch)
    training_history['train_loss'].append(round(avg_train_loss, 4))
    training_history['val_loss'].append(round(avg_val_loss, 4))
    training_history['val_accuracy'].append(round(val_accuracy, 2))

    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Time: {epoch_time:.0f}s", flush=True)

print("=" * 60, flush=True)
print("STAGE 8: Build unseen test set (72 examples) + evaluate", flush=True)
print("=" * 60, flush=True)

unseen_test_examples = [
    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'abeg make i see my account balance'},
    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'i wan know how much dey my account'},
    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'check my balance for me abeg'},
    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'wetin be my current balance'},
    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'my account balance dey worry me check am'},
    {'intent_label': 'balance_inquiry', 'context': 'abeg make i see my account balance', 'user_message': 'e don tey o wetin dey happen'},
    {'intent_label': 'balance_inquiry', 'context': 'i wan know how much dey my account', 'user_message': 'i never see the balance you talk about'},
    {'intent_label': 'balance_inquiry', 'context': 'check my balance for me abeg', 'user_message': 'make i know if e don update'},

    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'i send money but e no land'},
    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'my account don debit but transfer no complete'},
    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'the money wey i send no reach the person'},
    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'transfer wahala dey e never go'},
    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'una don take my money but transaction never successful'},
    {'intent_label': 'failed_transfer', 'context': 'i send money but e no land', 'user_message': 'the reference number na txn88213'},
    {'intent_label': 'failed_transfer', 'context': 'my account don debit but transfer no complete', 'user_message': 'abeg how long this go take to resolve'},
    {'intent_label': 'failed_transfer', 'context': 'the money wey i send no reach the person', 'user_message': 'e don tey the person never collect am'},

    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'i go atm but e no gree work'},
    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'the machine swallow my card'},
    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'atm debit me but no cash comot'},
    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'i fit no withdraw money for the atm'},
    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'atm dey give error message anytime i try'},
    {'intent_label': 'atm_issue', 'context': 'i go atm but e no gree work', 'user_message': 'i don already share the information'},
    {'intent_label': 'atm_issue', 'context': 'the machine swallow my card', 'user_message': 'nothing don change since'},
    {'intent_label': 'atm_issue', 'context': 'atm debit me but no cash comot', 'user_message': 'na lekki branch i dey talk about'},

    {'intent_label': 'loan_request', 'context': '', 'user_message': 'abeg i wan take loan from una bank'},
    {'intent_label': 'loan_request', 'context': '', 'user_message': 'wetin i go do to get loan'},
    {'intent_label': 'loan_request', 'context': '', 'user_message': 'i need money urgent make i apply for loan'},
    {'intent_label': 'loan_request', 'context': '', 'user_message': 'do una offer personal loans'},
    {'intent_label': 'loan_request', 'context': '', 'user_message': 'how una loan process take work'},
    {'intent_label': 'loan_request', 'context': 'abeg i wan take loan from una bank', 'user_message': 'no response since i apply'},
    {'intent_label': 'loan_request', 'context': 'wetin i go do to get loan', 'user_message': 'the amount wey i need na 150k'},
    {'intent_label': 'loan_request', 'context': 'i need money urgent make i apply for loan', 'user_message': 'i don submit my bvn already'},

    {'intent_label': 'card_issue', 'context': '', 'user_message': 'my atm card no gree work for any machine'},
    {'intent_label': 'card_issue', 'context': '', 'user_message': 'card dey decline anytime i wan pay'},
    {'intent_label': 'card_issue', 'context': '', 'user_message': 'i lost my card abeg block am sharp sharp'},
    {'intent_label': 'card_issue', 'context': '', 'user_message': 'pos machine no accept my card'},
    {'intent_label': 'card_issue', 'context': '', 'user_message': 'my debit card don stop working'},
    {'intent_label': 'card_issue', 'context': 'my atm card no gree work for any machine', 'user_message': 'make una resolve am quick quick'},
    {'intent_label': 'card_issue', 'context': 'card dey decline anytime i wan pay', 'user_message': 'the issue still dey persist'},
    {'intent_label': 'card_issue', 'context': 'i lost my card abeg block am sharp sharp', 'user_message': 'the last four digits na 7821'},

    {'intent_label': 'account_opening', 'context': '', 'user_message': 'abeg i wan start banking with una'},
    {'intent_label': 'account_opening', 'context': '', 'user_message': 'how i go do to get account number'},
    {'intent_label': 'account_opening', 'context': '', 'user_message': 'i wan create current account with una bank'},
    {'intent_label': 'account_opening', 'context': '', 'user_message': 'make una help me register account'},
    {'intent_label': 'account_opening', 'context': '', 'user_message': 'i wan be una customer how i go start'},
    {'intent_label': 'account_opening', 'context': 'abeg i wan start banking with una', 'user_message': 'wetin remain make i complete am'},
    {'intent_label': 'account_opening', 'context': 'how i go do to get account number', 'user_message': 'my documents never verify'},
    {'intent_label': 'account_opening', 'context': 'i wan create current account with una bank', 'user_message': 'i still never get response'},

    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'somebody don use my account do transaction wey i no do'},
    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'i think say my account don hack'},
    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'strange debit alert enter my phone'},
    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'abeg my account get fraudulent activity'},
    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'person don steal money from my account'},
    {'intent_label': 'fraud_report', 'context': 'somebody don use my account do transaction wey i no do', 'user_message': 'i dey worried say more money fit comot'},
    {'intent_label': 'fraud_report', 'context': 'i think say my account don hack', 'user_message': 'make una secure my account now now'},
    {'intent_label': 'fraud_report', 'context': 'strange debit alert enter my phone', 'user_message': 'the amount wey comot na 75k'},

    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'the app dey hang anytime i open am'},
    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'i no fit login to my mobile app'},
    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'una app no dey load at all'},
    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'the app keep showing error'},
    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'mobile app dey too slow to use'},
    {'intent_label': 'mobile_app_issue', 'context': 'the app dey hang anytime i open am', 'user_message': 'i don do everything you suggest'},
    {'intent_label': 'mobile_app_issue', 'context': 'i no fit login to my mobile app', 'user_message': 'the problem never solve'},
    {'intent_label': 'mobile_app_issue', 'context': 'una app no dey load at all', 'user_message': 'e still dey buggy'},

    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'abeg wetin be the news today'},
    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'make we talk about football small'},
    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'you get any good joke for me'},
    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'how the weather dey your side'},
    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'wetin be the best food for lagos'},
    {'intent_label': 'out_of_scope', 'context': 'abeg wetin be the news today', 'user_message': 'no wahala just dey gist me'},
    {'intent_label': 'out_of_scope', 'context': 'make we talk about football small', 'user_message': 'i still dey wait for your answer'},
    {'intent_label': 'out_of_scope', 'context': 'you get any good joke for me', 'user_message': 'abeg make i hear am now'},

    {'intent_label': 'balance_inquiry', 'context': '', 'user_message': 'i wan know wetin dey my account before i spend'},
    {'intent_label': 'failed_transfer', 'context': '', 'user_message': 'i try send something but e no work'},
    {'intent_label': 'atm_issue', 'context': '', 'user_message': 'the machine for my street no gree cooperate today'},
    {'intent_label': 'loan_request', 'context': '', 'user_message': 'i need extra cash wetin una fit offer me'},
    {'intent_label': 'card_issue', 'context': '', 'user_message': 'anytime i wan use am for shopping e no gree'},
    {'intent_label': 'account_opening', 'context': '', 'user_message': 'i wan start something with una as new customer'},
    {'intent_label': 'fraud_report', 'context': '', 'user_message': 'i see something for my statement wey i no recognize'},
    {'intent_label': 'mobile_app_issue', 'context': '', 'user_message': 'the thing for my phone no dey function well again'},
    {'intent_label': 'out_of_scope', 'context': '', 'user_message': 'abeg who be the best musician for naija right now'},
]

unseen_test_df = pd.DataFrame(unseen_test_examples)
unseen_test_df['user_message'] = unseen_test_df['user_message'].apply(clean_text)
unseen_test_df['user_message'] = unseen_test_df['user_message'].apply(normalize_pidgin_text)
unseen_test_df['user_message'] = unseen_test_df['user_message'].str.lower()
unseen_test_df['context'] = unseen_test_df['context'].apply(clean_text)
unseen_test_df['context'] = unseen_test_df['context'].apply(normalize_pidgin_text)
unseen_test_df['context'] = unseen_test_df['context'].str.lower()
unseen_test_df['encoded_label'] = label_encoder.transform(unseen_test_df['intent_label'])

unseen_input_ids, unseen_attention_masks = [], []
for _, row in unseen_test_df.iterrows():
    iid, am = tokenize_conversation_turn(row['context'], row['user_message'])
    unseen_input_ids.append(iid)
    unseen_attention_masks.append(am)

unseen_input_ids_tensor = torch.tensor(unseen_input_ids, dtype=torch.long)
unseen_attention_masks_tensor = torch.tensor(unseen_attention_masks, dtype=torch.long)
unseen_labels_tensor = torch.tensor(unseen_test_df['encoded_label'].tolist(), dtype=torch.long)

unseen_test_dataset = TensorDataset(unseen_input_ids_tensor, unseen_attention_masks_tensor, unseen_labels_tensor)
unseen_test_dataloader = DataLoader(unseen_test_dataset, batch_size=32, shuffle=False)

bert_model.eval()
all_true_labels, all_predicted_labels = [], []
with torch.no_grad():
    for batch in unseen_test_dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = bert_model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predicted_labels.extend(predictions.cpu().numpy().tolist())
        all_true_labels.extend(b_labels.cpu().numpy().tolist())

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

class_names = list(label_encoder.classes_)

metrics = {
    'test_samples': len(all_true_labels),
    'accuracy': accuracy_score(all_true_labels, all_predicted_labels),
    'precision_macro': precision_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0),
    'recall_macro': recall_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0),
    'f1_macro': f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0),
    'precision_weighted': precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0),
    'recall_weighted': recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0),
    'f1_weighted': f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0),
    'confusion_matrix': confusion_matrix(all_true_labels, all_predicted_labels).tolist(),
    'classification_report': classification_report(
        all_true_labels, all_predicted_labels, target_names=class_names,
        zero_division=0, output_dict=True),
    'class_names': class_names,
}

print(f"Unseen-test accuracy: {metrics['accuracy']*100:.2f}%", flush=True)

print("=" * 60, flush=True)
print("Saving model, tokenizer, and metadata", flush=True)
print("=" * 60, flush=True)

bert_model.save_pretrained(os.path.join(OUTPUT_DIR, 'model'))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'tokenizer'))

with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

RESPONSE_MAP = {
    ('balance_inquiry', 'initial'):
        "Abeg wait small, I go check your account balance for you now now. "
        "Your current balance don dey show for system.",
    ('balance_inquiry', 'follow_up'):
        "I don check again. Your balance still dey same as wetin I tell you before. "
        "You want make I send you statement of account?",
    ('failed_transfer', 'initial'):
        "Sorry for the wahala wey that transfer cause you. "
        "I go look am now. Abeg give me the transaction reference number.",
    ('failed_transfer', 'follow_up'):
        "I don see the transfer. E still dey pending processing. "
        "If e no clear within 24 hours, your money go return to your account.",
    ('atm_issue', 'initial'):
        "Sorry to hear say the ATM give you problem. "
        "Tell me the ATM location and wetin happen — whether card chop or money no comot.",
    ('atm_issue', 'follow_up'):
        "I don escalate the ATM complaint to our technical team. "
        "If your card dey inside machine, abeg block am now through the app for your safety.",
    ('loan_request', 'initial'):
        "You wan apply for loan? No wahala. "
        "You qualify for up to ₦500,000 based on your account history. "
        "You want make I start the process?",
    ('loan_request', 'follow_up'):
        "For the loan application, we go need your BVN, six months bank statement, "
        "and one valid ID. You get all these documents ready?",
    ('card_issue', 'initial'):
        "Sorry about the card issue. Tell me — na your card wey dem decline, "
        "card wey block, or you wan report lost card?",
    ('card_issue', 'follow_up'):
        "I don flag your card issue. If na lost card, I go block am immediately "
        "and request new card for you. New card go reach you within 5-7 working days.",
    ('account_opening', 'initial'):
        "Welcome! You wan open account with us — good choice. "
        "We get savings account, current account, and student account. "
        "Which one you dey consider?",
    ('account_opening', 'follow_up'):
        "To open the account, bring one valid ID (NIN or passport), "
        "one utility bill, and passport photograph to any of our branches. "
        "You fit also start online through our app.",
    ('fraud_report', 'initial'):
        "This na serious matter — I dey take am seriously. "
        "I don flag your account for security review immediately. "
        "Abeg describe wetin happen and when you notice the suspicious activity.",
    ('fraud_report', 'follow_up'):
        "I don escalate your fraud report to our security team. "
        "Your account don dey monitored. "
        "You go receive SMS update within the next 2 hours. "
        "No share your OTP or PIN with anybody.",
    ('mobile_app_issue', 'initial'):
        "Sorry say our app dey give you trouble. "
        "Wetin dey happen exactly — e no open, transaction dey fail, "
        "or you see error message?",
    ('mobile_app_issue', 'follow_up'):
        "Try these steps: clear app cache, make sure you get latest version, "
        "then restart your phone. If e still no work, I go reset your app session "
        "from our end right now.",
    ('out_of_scope', 'initial'):
        "Sorry o, I no fully understand wetin you talk. "
        "I be your banking assistant, so I fit help you better with things like "
        "account opening, balance check, transfer wahala, card or ATM issues, "
        "loan request, fraud report, or app problems. "
        "Abeg rephrase your question make e relate to one of these.",
    ('out_of_scope', 'follow_up'):
        "I still no understand wetin you dey try to ask. "
        "Make we face bank matter — you fit ask me about your balance, a transfer, "
        "your card, ATM, loan, fraud, account opening, or the mobile app.",
}

with open(os.path.join(OUTPUT_DIR, 'response_map.pkl'), 'wb') as f:
    pickle.dump(RESPONSE_MAP, f)

config = {
    'model_name': MODEL_NAME,
    'max_length': MAX_LENGTH,
    'num_labels': num_intent_labels,
    'class_names': class_names,
    'train_samples': len(train_indices),
    'val_samples': len(val_indices),
    'num_epochs': NUM_EPOCHS,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(training_history, f, indent=2)

print("=" * 60, flush=True)
print("DONE — saved to", OUTPUT_DIR, flush=True)
print("=" * 60, flush=True)
