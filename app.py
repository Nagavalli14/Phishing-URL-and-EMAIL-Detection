import sys
import os

# --- 1. COMPATIBILITY SHIM FOR KERAS 3 ---
# This fixes "ModuleNotFoundError: No module named 'keras.preprocessing.text'"
try:
    import keras
    import keras.src.legacy.preprocessing.text as text
    sys.modules['keras.preprocessing.text'] = text
except (ImportError, AttributeError):
    pass

import streamlit as st
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM

# --- 2. THE LSTM KEYWORD FIX ---
# This custom class ignores the 'time_major' argument that crashes Keras 3
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove the problematic argument
        super().__init__(*args, **kwargs)

# -----------------------------
# 3. LOAD MODELS
# -----------------------------
@st.cache_resource
def load_all_models():
    # Load ML components
    url_ml = joblib.load("models/url_ml.pkl")
    email_ml = joblib.load("models/email_ml.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    tokenizer = joblib.load("models/tokenizer_tf.pkl")
    
    # Load DL Models with the custom LSTM patch
    # This specifically fixes the "Unrecognized keyword arguments passed to LSTM" error
    custom_objects = {'LSTM': CompatibleLSTM}
    
    url_dl = load_model("models/url_dl.h5", custom_objects=custom_objects, compile=False)
    email_dl = load_model("models/email_dl.h5", custom_objects=custom_objects, compile=False)
    
    return url_ml, email_ml, tfidf, tokenizer, url_dl, email_dl

# Initialize models
try:
    url_ml, email_ml, tfidf, tokenizer, url_dl, email_dl = load_all_models()
except Exception as e:
    st.error(f"Critical Error Loading Models: {e}")
    st.info("Try running: pip install tensorflow==2.15.0")
    st.stop()

# -----------------------------
# 4. FEATURE EXTRACTION (URL)
# -----------------------------
def extract_url_features(url):
    features = {}
    features['UsingIP'] = 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
    features['LongURL'] = 1 if len(url) > 75 else 0
    features['ShortURL'] = 1 if re.search(r'tinyurl|bit\.ly|goo\.gl', url) else 0
    features['Symbol@'] = 1 if '@' in url else 0
    features['Redirecting//'] = 1 if urlparse(url).path.count('//') > 0 else 0
    features['PrefixSuffix-'] = 1 if '-' in urlparse(url).netloc else 0
    features['SubDomains'] = 1 if urlparse(url).netloc.count('.') > 2 else 0
    features['HTTPS'] = 1 if urlparse(url).scheme == 'https' else 0
    features['DomainRegLen'] = 1 if len(urlparse(url).netloc) < 10 else 0
    
    # Fill remaining features with 0 as expected by your trained model
    other_features = ['Favicon','Port','HTTPSdomain','RequestURL','URLofAnchor',
                      'LinksInScriptTags','SFH','SubmittingToEmail','AbnormalURL','Redirect',
                      'OnMouseOver','RightClick','PopUpWidnow','Iframe','AgeofDomain','DNSRecord',
                      'WebsiteTraffic','PageRank','GoogleIndex','LinksPointingToPage','StatsReport']
    for feat in other_features:
        features[feat] = 0
    return features

# -----------------------------
# 5. PREDICTION LOGIC
# -----------------------------
def predict_url(features_dict):
    features = np.array([list(features_dict.values())])
    ml_prob = url_ml.predict_proba(features)[0][1]
    dl_prob = url_dl.predict(features)[0][0]
    final_prob = (ml_prob + dl_prob)/2
    
    verdict = "PHISHING URL 🚨" if ml_prob > 0.5 or dl_prob > 0.5 else "SAFE URL ✅"
    return {
        "Verdict": verdict,
        "ML Confidence": f"{ml_prob*100:.2f}%",
        "DL Confidence": f"{dl_prob*100:.2f}%",
        "Final Risk": f"{final_prob*100:.2f}%"
    }

feature_names = np.array(tfidf.get_feature_names_out())

def explain_email(text, top_n=5):
    vec = tfidf.transform([text])
    scores = vec.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return feature_names[top_indices].tolist()

def predict_email_zero_trust_keywords(text):
    ml_vec = tfidf.transform([text])
    ml_prob = email_ml.predict_proba(ml_vec)[0][1]
    
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=200)
    dl_prob = email_dl.predict(pad_seq)[0][0]
    
    keywords = explain_email(text, top_n=5)
    critical_words = ['urgent', 'verify', 'bank', 'account', 'click', 'password', 
                      'suspend', 'login', 'immediately']
    
    verdict = "PHISHING EMAIL 🚨" if (ml_prob > 0.5 or dl_prob > 0.5 or 
                                     any(w in text.lower() for w in critical_words)) else "SAFE EMAIL ✅"
    
    return {
        "Verdict": verdict,
        "ML Prob": f"{ml_prob*100:.2f}%",
        "DL Prob": f"{dl_prob*100:.2f}%",
        "Key Tokens": keywords
    }

# -----------------------------
# 6. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Phishing Shield", layout="wide")
st.title("🛡️ Zero-Trust Phishing Detector")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    option = st.selectbox("Scan Type", ["URL Analysis", "Email Analysis"])
    
    if option == "URL Analysis":
        url_in = st.text_input("Enter URL:")
        if url_in:
            res = predict_url(extract_url_features(url_in))
            with col2:
                if "PHISHING" in res["Verdict"]: st.error(res["Verdict"])
                else: st.success(res["Verdict"])
                st.write(res)

    else:
        email_in = st.text_area("Enter Email Body:", height=250)
        if email_in:
            res = predict_email_zero_trust_keywords(email_in)
            with col2:
                if "PHISHING" in res["Verdict"]: st.error(res["Verdict"])
                else: st.success(res["Verdict"])
                st.write(f"**Top Suspicious Terms:** {', '.join(res['Key Tokens'])}")
                st.write(f"**ML Score:** {res['ML Prob']}")
                st.write(f"**DL Score:** {res['DL Prob']}")