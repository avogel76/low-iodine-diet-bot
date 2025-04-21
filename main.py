import streamlit as st
import difflib
import os
from sentence_transformers import SentenceTransformer, util
import torch
import pickle

# --- הגדרות ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDINGS_PATH = 'embeddings/base.pkl'
TEXT_PATH = 'knowledge/base.txt'

# --- טען טקסט ---
def load_text():
    with open(TEXT_PATH, 'r', encoding='utf-8') as f:
        return f.read().split('\n')

# --- טען מודל ווקטורים ---
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        return None, None
    with open(EMBEDDINGS_PATH, 'rb') as f:
        return pickle.load(f)

# --- חפש תשובה ---
def find_best_answer(question, texts, embeddings, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    if best_score < 0.4:
        return None
    return texts[best_idx]

# --- Streamlit ---
st.set_page_config(page_title="בוט דיאטה דלת יוד", page_icon="🥗", layout="centered")
st.title("🤖 בוט דיאטה דלת יוד")
st.write("שאלו אותי על מאכלים, שתייה, מתכונים או כל דבר אחר שקשור לדיאטה לפני טיפול ביוד רדיואקטיבי.")

user_input = st.text_input("מה ברצונך לשאול?", "")

if user_input:
    st.write("מחפש תשובה…")
    model = SentenceTransformer(MODEL_NAME)
    texts = load_text()
    embeddings, original_texts = load_embeddings()

    if embeddings is None:
        st.error("לא נמצאו וקטורים. אנא הרץ סקריפט לאינדוקס הטקסטים (ראה README).")
    else:
        answer = find_best_answer(user_input, original_texts, embeddings, model)
        if answer:
            st.success(answer)
        else:
            st.warning("לא מצאתי תשובה מדויקת במסמך. אפשר לבדוק מול מקורות נוספים או להתייעץ עם רופא.")

st.markdown("---")
st.caption("נבנה בהתבסס על דף ההנחיה של ד"ר איל רובינשטוק מוויקירפואה")