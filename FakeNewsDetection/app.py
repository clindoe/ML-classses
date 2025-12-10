import pickle
import string

import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==============================
# 1. page config + simple CSS
# ==============================

st.set_page_config(
    page_title="Fake News / Spam Detector",
    page_icon="📰",
    layout="wide"
)

custom_css = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        margin-top: 1rem;
    }
    .result-good {
        background-color: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.6);
    }
    .result-bad {
        background-color: rgba(231, 76, 60, 0.12);
        border: 1px solid rgba(231, 76, 60, 0.6);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ==============================
# 2. preprocessing (match training)
# ==============================

english_stopwords = set(stopwords.words("english"))
english_punctuation = string.punctuation
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""

    # remove punctuation
    chars = [ch for ch in text if ch not in english_punctuation]
    clean_text = "".join(chars)

    # lowercase
    clean_text = clean_text.lower()

    # remove stopwords
    words = clean_text.split()
    words = [w for w in words if w not in english_stopwords]

    return " ".join(words)


def lemmatize_text(text: str) -> str:
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())


def full_preprocess(text: str) -> str:
    return lemmatize_text(preprocess_text(text))


# ==============================
# 3. load model + vectorizer
# ==============================

@st.cache_resource
def load_model():
    with open("logistic_regression_tfidf.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["vectorizer"]


model, vectorizer = load_model()

# change names if your labels mean something else
CLASS_NAMES = {
    0: "Fake / Spam",
    1: "Real / Ham"
}


# ==============================
# 4. sidebar
# ==============================

st.sidebar.title("About this app")
st.sidebar.markdown(
    """
This app loads a **TF–IDF + Logistic Regression** model
you trained in your notebook and uses it to classify text.

- Preprocessing: punctuation removal, lowercasing, stopwords, lemmatization  
- Vectorizer: TF–IDF  
- Model: Logistic Regression (saved in `logistic_regression_tfidf.pkl`)
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick examples")

example_texts = {
    "Short spam example": "Congratulations you have won a free iPhone Click this link now to claim your prize",
    "Short real news example": "The government announced new economic measures to support small businesses affected by inflation.",
    "Empty": ""
}

example_choice = st.sidebar.selectbox(
    "Insert an example",
    list(example_texts.keys())
)

if st.sidebar.button("Use example"):
    st.session_state["example_text"] = example_texts[example_choice]


# ==============================
# 5. main layout
# ==============================

st.markdown(
    '<div class="main-title">Fake News / Spam Text Classifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Paste a news article or message and the model will try to decide if it is more likely fake / spam or real / ham.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([2.5, 1.5])

with left_col:
    default_text = st.session_state.get("example_text", "")

    user_text = st.text_area(
        "Input text",
        value=default_text,
        height=260,
        placeholder="Paste your article, email, or message here...",
    )

    predict_button = st.button("Run prediction")

with right_col:
    st.markdown("### Prediction")

    if predict_button:
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            clean = full_preprocess(user_text)
            vec = vectorizer.transform([clean])

            pred = int(model.predict(vec)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(vec)[0, 1])

            label_text = CLASS_NAMES.get(pred, str(pred))

            if pred == 1:
                css_class = "result-card result-good"
                icon = "🟢"
            else:
                css_class = "result-card result-bad"
                icon = "🔴"

            st.markdown(
                f"""
                <div class="{css_class}">
                    <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.2rem;">
                        {icon} {label_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if prob is not None:
                st.markdown("#### Confidence")
                st.progress(prob)
                st.write(
                    f"Estimated probability of class 1 (Real / Ham): **{prob:.3f}**"
                )

            with st.expander("Preprocessed text"):
                st.write(clean)
    else:
        st.info("Paste some text and click **Run prediction** to see the result.")
