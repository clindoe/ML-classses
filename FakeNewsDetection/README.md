# 📰 Fake News / Spam Detection App

**Deployed on Streamlit Cloud** | **Live Demo:** [Open App](https://fakenewsdetectionbyajmainadil.streamlit.app/)

A machine learning-powered web application that classifies text as **Fake/Spam** or **Real/Ham** using Natural Language Processing and Logistic Regression.

---

## 🎯 Project Overview

This project provides an end-to-end solution for detecting fake news and spam messages using:
- **NLP Preprocessing:** Punctuation removal, stopword elimination, lemmatization
- **Feature Engineering:** TF-IDF vectorization
- **Machine Learning:** Logistic Regression classifier
- **Production Deployment:** Streamlit web interface

---

## 📊 Dataset Information

| Column | Description |
|--------|-------------|
| `title` | News article or message title |
| `text` | Full text content to classify |
| `label` | 0 = Fake/Spam, 1 = Real/Ham |

The dataset was cleaned, null values handled, and text normalized for optimal model performance.

---

## 🔧 Text Preprocessing Pipeline

1. **Remove Punctuation** - Strips special characters
2. **Lowercase Conversion** - Normalizes text case
3. **Stopword Removal** - Filters common English words
4. **Lemmatization** - Reduces words to base forms using WordNetLemmatizer
5. **Part-of-Speech (POS) Tagging** - Filters adjectives, verbs, and other important parts of speech
6. **Tokenization** - Splits text into words for processing

This ensures the model trains on clean, normalized input data with linguistically meaningful tokens.

---

## 🤖 Model Training & Selection

Multiple models were evaluated on the training dataset:

| Model | Performance | Status |
|-------|-------------|--------|
| Multinomial Naive Bayes | Good | Evaluated |
| Bernoulli Naive Bayes | Good | Evaluated |
| Logistic Regression | **Best** | ✅ Selected |

**Training Configuration:**
- Train-Test Split: 80% / 20%
- Vectorizer: TF-IDF (fit on training data only)
- Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report, ROC-AUC

**Why Logistic Regression?**
- Highest accuracy and generalization
- Best ROC-AUC performance
- Optimal for binary classification
- Provides confidence probabilities

---

## 📁 Project Structure

```
FakeNewsDetection/
├── app.py                           # Streamlit web interface
├── FakeNewsDetection.ipynb         # Model training notebook
├── logistic_regression_tfidf.pkl   # Pre-trained model + vectorizer
├── text_dataset.csv                # Training dataset
├── requirements.txt                # Python dependencies
├── nltk.txt                        # NLTK data requirements
├── .streamlit/config.toml          # Streamlit configuration
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/clindoe/ML-classses.git
   cd FakeNewsDetection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:** Navigate to `http://localhost:8501`

---

## 📦 Dependencies

- **streamlit** - Web framework
- **nltk** - Natural Language Toolkit
- **scikit-learn** - Machine Learning library
- **numpy** - Numerical computing
- **pandas** - Data manipulation

See `requirements.txt` for specific versions.

---

## 💻 How to Use the App

1. **Paste Text:** Enter a news article, email, or message in the text area
2. **Click Predict:** Press "Run prediction" button
3. **View Results:**
   - Classification: **Real/Ham** (🟢) or **Fake/Spam** (🔴)
   - Confidence Score: Probability of being real/ham
   - Preprocessed Text: See the cleaned version used by the model

### Try Examples
Quick-start with pre-loaded examples:
- Short spam example
- Short real news example
- Empty input

---

## 📈 Model Performance

The Logistic Regression model achieves:
- High accuracy on test data
- Good generalization capability
- Confidence-calibrated predictions
- Fast inference time

---

## 🔐 Model Serialization

The trained model and vectorizer are saved together in a single pickle file:

```python
{
    "model": LogisticRegressionClassifier,
    "vectorizer": TfidfVectorizer
}
```

This ensures consistency between preprocessing and prediction during inference.

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Select branch and app file: `FakeNewsDetection/app.py`
5. Deploy!

Your app will be available at: `https://[username]-[projectname].streamlit.app/`

---

## 📝 Files Explanation

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI and inference logic |
| `FakeNewsDetection.ipynb` | Model training notebook (analysis, EDA, model development) |
| `logistic_regression_tfidf.pkl` | Serialized model and vectorizer |
| `text_dataset.csv` | Training dataset |
| `requirements.txt` | Python package dependencies |
| `nltk.txt` | NLTK data to download (stopwords, wordnet, etc.) |

---

## 🐛 Troubleshooting

**Issue:** NLTK data not found
- **Solution:** The app automatically downloads required NLTK data on startup

**Issue:** Model file not found
- **Solution:** Ensure `logistic_regression_tfidf.pkl` is in the same directory as `app.py`

**Issue:** Slow predictions
- **Solution:** Predictions are cached in Streamlit; first run loads the model, subsequent runs are instant

---

## 📚 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Classification](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## 📄 License

This project is part of the ML-classses learning repository.

---

## 👨‍💻 Author

Created as part of machine learning coursework.

---

## ⭐ Features

✅ Real-time text classification  
✅ Confidence probability scores  
✅ Preprocessing transparency (view cleaned text)  
✅ Quick example templates  
✅ Clean, intuitive UI  
✅ Production-ready deployment  
✅ Mobile-friendly interface
