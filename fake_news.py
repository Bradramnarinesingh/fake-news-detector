import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import joblib

# -------------------------------------------------------------------
# 1. Load and Combine Datasets
# -------------------------------------------------------------------
fake_df = pd.read_csv('fake.csv')   # CSV file of fake news
true_df = pd.read_csv('true.csv')   # CSV file of real news

# Label: 0 for Fake, 1 for Real
fake_df['label'] = 0
true_df['label'] = 1

# Merge both datasets
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Keep only text + label columns
df = df[['text', 'label']]

# Shuffle for unbiased training
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preview dataset
print(df.head())

# -------------------------------------------------------------------
# 2. Download Stopwords & Setup
# -------------------------------------------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------------------------------------------
# 3. Clean Text Function
# -------------------------------------------------------------------
def clean_text(text):
    """
    Cleans the input text by:
      1. Removing HTML tags
      2. Removing special characters (but keeping numbers)
      3. Calculating the proportion of excessive punctuation ("!!!"/"???")
      4. Removing stopwords (case-insensitive, preserving capitalization in final text if desired)
      5. Removing extra spaces
    
    Returns:
      cleaned_text (str)
      excessive_punctuation_ratio (float)
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Count multiple punctuation "!!!" or "???"
    excessive_punct_count = len(re.findall(r'!{2,}|\?{2,}', text))
    total_punct_count = len(re.findall(r'[!?]', text))
    if total_punct_count == 0:
        excessive_punct_ratio = 0
    else:
        excessive_punct_ratio = excessive_punct_count / total_punct_count
    
    # Remove multiple punctuation but keep single exclamation/question marks
    text = re.sub(r'!{2,}|\?{2,}', '', text)
    
    # Remove special characters (but keep numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove stopwords (case-insensitive)
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_words).strip()
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text, excessive_punct_ratio

# -------------------------------------------------------------------
# 4. Clean Text & Vectorize (TF-IDF)
# -------------------------------------------------------------------
# Apply clean_text function to each article
df[['clean_text', 'excessive_punctuation_ratio']] = df['text'].apply(
    lambda x: pd.Series(clean_text(x))
)

# Convert cleaned text to TF-IDF vectors (up to 5000 features)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_text']).toarray()

# Combine TF-IDF features with the punctuation ratio
punct_feature = df['excessive_punctuation_ratio'].values.reshape(-1, 1)
X = np.hstack((X_tfidf, punct_feature))

# Labels: 0 = Fake, 1 = Real
y = df['label'].values

print('Features shape:', X.shape)

# -------------------------------------------------------------------
# 5. Train/Test Split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training data: {X_train.shape}, Testing data: {X_test.shape}')

# -------------------------------------------------------------------
# 6. Build Neural Network Model
# -------------------------------------------------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # 1st layer is an explicit Input
    Dense(512, activation='relu'),
    Dropout(0.6),     # High dropout to reduce overfitting
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(1, activation='sigmoid')    # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------------------------------------------
# 7. Train the Model
# -------------------------------------------------------------------
# (Reduce epochs to 3 if overfitting remains an issue)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

print("Model training complete!")

# -------------------------------------------------------------------
# 8. Evaluate the Model
# -------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# -------------------------------------------------------------------
# 9. Prediction Function
# -------------------------------------------------------------------
def predict_fake_news(text):
    """
    Predicts if a given news article is FAKE or REAL.
    Returns (label, confidence).
    """
    cleaned_text, punct_ratio = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text]).toarray()
    punct_vec = np.array([[punct_ratio]])
    
    final_features = np.hstack((text_vec, punct_vec))
    pred = model.predict(final_features)[0][0]
    
    label = "FAKE NEWS!" if pred < 0.5 else "REAL NEWS!"
    confidence = float(pred)
    return label, confidence

# -------------------------------------------------------------------
# 10. Quick Test
# -------------------------------------------------------------------
test_article = (
    "BREAKING: Government officials have confirmed that the president "
    "will be stepping down from office next week."
)
label, conf = predict_fake_news(test_article)
print(f"Sample Prediction: {label}, Confidence: {conf:.2f}")

# -------------------------------------------------------------------
# 11. Test with Sample Article
# -------------------------------------------------------------------
news_article = (
    "BREAKING: Government officials have confirmed that the president "
    "will be stepping down from office next week."
)
label, confidence = predict_fake_news(news_article)
print(f'Prediction: {label}, Confidence: {confidence:.4f}')

# -------------------------------------------------------------------
# 12. Additional Sample Testing
# -------------------------------------------------------------------
samples = [
    "Official sources confirm that NASA discovered a new planet beyond Pluto.",
    "Study reveals coffee cures every disease known to humankind!",
    "President calls for new elections amid corruption scandal.",
    "Scientists announce cure for cancer available for free next month.",
    "The Earth is flat and NASA has been hiding the truth for years."
]

print("\n--- Additional Sample Predictions ---")
for article in samples:
    label, conf = predict_fake_news(article)
    print(f"Article: {article}\nPrediction: {label}, Confidence: {conf:.4f}\n")