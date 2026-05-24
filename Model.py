import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# 2. Preprocessing setup
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Base Models & Pipeline
ensemble = VotingClassifier(
    estimators=[
        ("mnb", MultinomialNB()),
        ("svc", SVC(probability=True, random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
    ]
)

pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", ensemble)])

# 4. Optimize Hyperparameters
param_grid = {
    "classifier__mnb__alpha": [1.0, 5.0],
    "classifier__svc__C": [1, 10],
    "classifier__svc__kernel": ["linear"],
    "classifier__rf__n_estimators": [100, 200],
    "classifier__voting": ["soft"],
}

print("Training and optimizing model... Please wait.")
cls = RandomizedSearchCV(
    pipe, param_distributions=param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1
)
cls.fit(X_train, y_train)

# 5. Evaluate
print(f"Test Accuracy: {cls.score(X_test, y_test) * 100:.2f}%")

# 6. SAVE THE MODEL
pickle.dump(cls, open("obesity_model.pkl", "wb"))
print("Success! Model saved as 'obesity_model.pkl'")
