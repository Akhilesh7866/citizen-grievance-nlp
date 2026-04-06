import pickle

# 👉 IMPORT your trained objects from your notebook or recreate them here
# Example:
# from your_training_file import tfidf_vectorizer, model

# TEMP placeholder (REMOVE when using real ones)
# tfidf_vectorizer = ...
# model = ...

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully!")
