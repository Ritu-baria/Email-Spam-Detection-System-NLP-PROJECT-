# Email Spam Detection using NLP.py
# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download stopwords (only once)
nltk.download('stopwords')

# Load dataset from Kaggle
df = pd.read_csv('spam.csv')
df.columns = ['label', 'text']

# Preprocessing
ps = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)           # Remove symbols
    text = text.lower()                             # Lowercase
    text = text.split()                             # Tokenize
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]  # Remove stopwords + stem
    return ' '.join(text)

df['cleaned'] = df['text'].apply(preprocess)

# Convert labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])   # Vectorized features
y = df['label_num']                   # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocess same way
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Predict new message
def predict_spam(msg):
    msg_clean = preprocess(msg)
    vect = cv.transform([msg_clean])
    pred = model.predict(vect)
    return "Spam" if pred[0] == 1 else "Not Spam"

# Example
print(predict_spam("Free entry in a contest! Call now!"))
