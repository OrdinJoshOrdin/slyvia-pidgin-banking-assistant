# ============================================================
# Naija Pidgin Banking Assistant — Streamlit App
#
# A chat interface backed by a fine-tuned BERT
# (bert-base-uncased) intent classifier for Nigerian Pidgin
# banking customer-service conversations.
# ============================================================

import os
import re
import json
import pickle

import torch
import torch.nn.functional as F
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models', 'slyvia_pidgin_bert')

st.set_page_config(
    page_title="Naija Pidgin Banking Assistant",
    page_icon="💬",
    layout="centered",
)

# ------------------------------------------------------------
# Force a clean, light interface regardless of system theme
# ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stSidebar"], [data-testid="stChatMessage"], .stMarkdown, .stButton > button,
    [data-testid="stChatInput"], textarea, input, p, div, span, h1, h2, h3, h4, label {
        font-family: 'Poppins', sans-serif !important;
    }

    span[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"],
    .material-symbols-rounded, .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] div,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        color: #000000 !important;
        background-color: transparent !important;
    }

    code {
        background-color: transparent !important;
        color: inherit !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }

    [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
        font-size: 0 !important;
    }
    [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"]::before,
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"]::before {
        content: "menu";
        font-family: 'Material Symbols Rounded' !important;
        font-size: 1.5rem;
        color: #000000;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #ffffff;
        color: #1f2937;
    }
    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #f7f9fc;
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        border: 1px solid #eef0f3;
    }
    .metric-card {
        background-color: #f7f9fc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #d0d7de;
        background-color: #ffffff;
        color: #000000 !important;
    }
    .stButton>button:hover {
        background-color: #1f2937 !important;
        border-color: #1f2937 !important;
        color: #ffffff !important;
    }
    .stButton>button[kind="primary"] {
        background-color: #0d6efd;
        border-color: #0d6efd;
        color: #ffffff !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #0b5ed7 !important;
        border-color: #0b5ed7 !important;
        color: #ffffff !important;
    }
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .app-header h1 {
        margin-bottom: 0.1rem;
    }
    .app-header p {
        color: #6b7280;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Load model + tokenizer + metadata (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the fine-tuned BERT model...")
def load_assets():
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'response_map.pkl'), 'rb') as f:
        response_map = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'tokenizer'))
    model = BertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, 'model'))

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    return config, metrics, label_encoder, response_map, tokenizer, model, device


if not os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
    st.error(
        "Model files not found. Please run `train_and_save_slyvia_model.py` "
        "first to train and save the model."
    )
    st.stop()

config, metrics, label_encoder, response_map, tokenizer, model, device = load_assets()
MAX_LENGTH = config['max_length']


# ------------------------------------------------------------
# Preprocessing + prediction helpers (mirrors the notebook)
# ------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_pidgin_text(text):
    text = str(text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.replace(' dnt ', ' dont ')
    text = text.replace(' wnt ', ' wont ')
    text = text.replace(' cnt ', ' cant ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text):
    return normalize_pidgin_text(clean_text(text)).lower()


def tokenize_conversation_turn(context, message, max_length=MAX_LENGTH):
    if context.strip() == '':
        encoding = tokenizer(message, padding='max_length', truncation=True,
                              max_length=max_length, return_attention_mask=True)
    else:
        encoding = tokenizer(context, message, padding='max_length', truncation=True,
                              max_length=max_length, return_attention_mask=True)
    return encoding['input_ids'], encoding['attention_mask']


def predict_intent_and_response(new_message, conversation_history):
    cleaned_message = preprocess(new_message)

    if len(conversation_history) == 0:
        context = ''
        conversation_stage = 'initial'
    else:
        context = preprocess(conversation_history[-1])
        conversation_stage = 'follow_up'

    input_ids, attention_mask = tokenize_conversation_turn(context, cleaned_message)
    input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)

    probabilities = F.softmax(outputs.logits, dim=1)
    confidence_score = probabilities.max().item() * 100
    predicted_class = outputs.logits.argmax(dim=1).item()
    intent_label = label_encoder.inverse_transform([predicted_class])[0]

    bot_response = response_map.get(
        (intent_label, conversation_stage),
        "Sorry, I no understand fully. Abeg explain again make I help you better."
    )

    return intent_label, round(confidence_score, 2), bot_response


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'user_history' not in st.session_state:
    st.session_state.user_history = []
if 'conversation_ended' not in st.session_state:
    st.session_state.conversation_ended = False


def reset_conversation():
    st.session_state.chat_messages = []
    st.session_state.user_history = []
    st.session_state.conversation_ended = False


# ------------------------------------------------------------
# Sidebar — model info
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h3><span class="material-symbols-rounded" '
        'style="vertical-align:middle; font-size:1.3rem; margin-right:0.3rem;">'
        'smart_toy</span>About this model</h3>',
        unsafe_allow_html=True,
    )
    st.markdown("**Model Name:** BERT")
    st.markdown("**Architecture:** Transformer Encoder")
    st.markdown(f"**Trained on:** {config['train_samples']:,} conversation turns")
    st.markdown(f"**Unseen-test accuracy:** {metrics['accuracy'] * 100:.1f}%")
    st.markdown("**Recognized intents:**")
    for c in config['class_names']:
        st.markdown(f"- {c.replace('_', ' ').title()}")
    st.markdown("---")
    st.caption(
        "This assistant understands Nigerian Pidgin English and routes "
        "your message to the right banking support flow."
    )


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Naija Pidgin Banking Assistant</h1>
        <p>Chat with our AI for Pidgin - Use that textbox for down write wetin you wan write</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------
for msg in st.session_state.chat_messages:
    avatar = "🧑" if msg['role'] == 'user' else "🤖"
    with st.chat_message(msg['role'], avatar=avatar):
        st.markdown(msg['content'])

if not st.session_state.conversation_ended:
    user_input = st.chat_input("Type your message in Pidgin...")
    if user_input and user_input.strip():
        st.session_state.chat_messages.append({'role': 'user', 'content': user_input})

        intent, confidence, bot_response = predict_intent_and_response(
            user_input, st.session_state.user_history
        )
        st.session_state.user_history.append(user_input)

        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': bot_response,
            'intent': intent,
            'confidence': confidence,
        })
        st.rerun()

st.markdown("---")

# ------------------------------------------------------------
# Conversation controls
# ------------------------------------------------------------
if not st.session_state.conversation_ended:
    if st.session_state.chat_messages:
        if st.button("End Conversation", use_container_width=True, type="primary", icon=":material/flag:"):
            st.session_state.conversation_ended = True
            st.rerun()
else:
    st.success("Conversation ended. Thanks for chatting with us!")

    if st.button("Start New Conversation", use_container_width=True, icon=":material/refresh:"):
        reset_conversation()
        st.rerun()
