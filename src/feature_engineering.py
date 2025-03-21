# Generate TF-IDF + N-gram features from preprocessed prompts

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv", sep='\t')

# Combine all prompt types into a single column for vectorization
df["combined_prompts"] = df["vanilla"].fillna('') + " " + df["adversarial"].fillna('')

# Initialize TF-IDF Vectorizer with N-gram support
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),      # Unigrams, Bigrams, Trigrams
    max_features=10000,     # Limit to top 10,000 features
    sublinear_tf=True,      # Apply sublinear term frequency scaling
    stop_words='english'    
)

# Fit and transform text data into TF-IDF features
X_tfidf = vectorizer.fit_transform(df["combined_prompts"])

# Save TF-IDF matrix and vectorizer
joblib.dump(X_tfidf, "data/tfidf_features.pkl")
joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

# Preview top N features
feature_names = vectorizer.get_feature_names_out()
print("Sample TF-IDF Features:", feature_names[:20])
