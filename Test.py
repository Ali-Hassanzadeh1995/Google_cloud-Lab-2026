import pandas as pd
import pickle

# 1. Load the trained model from disk
loaded_model = pickle.load(open("obesity_model.pkl", "rb"))

# 2. Simulate getting input from a user (e.g., from a web form or terminal)
user_input = {
    "Gender": "Male",
    "Age": 31,
    "Height": 1.84,
    "Weight": 110,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2.0,
    "NCP": 2.0,
    "CAEC": "Sometimes",
    "SMOKE": "yes",
    "CH2O": 3.0,
    "SCC": "no",
    "FAF": 1.0,
    "TUE": 3.0,
    "CALC": "Sometimes",
    "MTRANS": "Automobile",
}

# 3. Convert the single user input dictionary into a pandas DataFrame
user_df = pd.DataFrame([user_input])

# 4. Predict the Obesity Level!
prediction = loaded_model.predict(user_df)

print(f"Predicted Obesity Level: {prediction[0]}")

# Optional: Because we used 'soft' voting, you can also see the probability
# of the prediction (how confident the model is)
probabilities = loaded_model.predict_proba(user_df)
print(f"Prediction Confidence Probabilities:\n{probabilities}")
