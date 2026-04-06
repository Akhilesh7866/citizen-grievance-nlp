import pickle

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_grievance(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]


def get_priority(department):
    dept = department.lower()

    if "police" in dept or "emergency" in dept:
        return "High"
    elif "water" in dept or "electricity" in dept:
        return "Medium"
    else:
        return "Low"


def format_output(text):
    try:
        prediction = predict_grievance(text)

        # Confidence score (if available)
        try:
            text_vector = vectorizer.transform([text])
            confidence = max(model.predict_proba(text_vector)[0])
            confidence = round(confidence, 3)
        except:
            confidence = "N/A"

        response = {
            "input": {
                "text": text
            },
            "output": {
                "predicted_department": prediction,
                "confidence_score": confidence,
                "priority": get_priority(prediction)
            },
            "meta": {
                "model_used": "Random Forest with TF-IDF",
                "status": "success"
            },
            "message": f"Your complaint has been categorized under '{prediction}' department."
        }

    except Exception as e:
        response = {
            "input": {
                "text": text
            },
            "error": str(e),
            "meta": {
                "status": "failed"
            }
        }

    return response


# Local test
if __name__ == "__main__":
    sample = "Street lights are not working in my area"
    print(format_output(sample))
