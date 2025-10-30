
import shap
import sys
import re
import string
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# loading csv file 

CSV_PATH = "ks-projects-201612.csv"

df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1", low_memory=False)
print("Initial Shape:", df.shape)

# Strip spaces, lowercase, replace spaces with underscores
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Dropping empty "unnamed" extra columns if present
drop_cols = [c for c in df.columns if c.startswith("unnamed")]
if drop_cols:
    df = df.drop(columns=drop_cols)

# 2) Sanity checks for required columns

required_text_cols = []  # we'll assemble a text field from any of these that exist
for c in ["name", "category", "main_category", "country", "currency"]:
    if c in df.columns:
        required_text_cols.append(c)

if "state" not in df.columns:
    print("[ERROR] Could not find 'state' column even after normalizing headers.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

if not required_text_cols:
    print("[ERROR] Could not find any text columns among: name, category, main_category, country, currency")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

# 3) Filter to binary outcomes (successful/failed) & clean target

df["state"] = df["state"].astype(str).str.strip().str.lower()
df = df[df["state"].isin(["successful", "failed"])].copy()
df["target"] = (df["state"] == "successful").astype(int)


# 4) Build a single text field

def clean_text_basic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)              # remove urls
    text = text.translate(str.maketrans("", "", string.punctuation))  # rm punctuation
    text = re.sub(r"\d+", " ", text)                  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text

# Combine any available text columns into 'text'
df["text"] = (
    df[required_text_cols]
    .fillna("")
    .agg(" ".join, axis=1)
    .apply(clean_text_basic)
)

# Remove rows with very short/empty text (less than 3 tokens)
df = df[df["text"].str.split().str.len() >= 3].copy()

# 5) handling Numeric features 

numeric_candidates = [
    "goal",              # present in this CSV
    "usd_pledged",       # in this CSV as 'usd pledged' -> normalized to 'usd_pledged'
    "usd_pledged_real",  # present in some variants
    "usd_goal_real",     # present in some variants
    "pledged",           # original currency pledged
    "backers"
]
numeric_features = [c for c in numeric_candidates if c in df.columns]

#  converting to numeric
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Removing impossible goals (<= 0) if 'goal' exists
if "goal" in df.columns:
    df = df[df["goal"] > 0].copy()

# Fill remaining NaNs in numeric with median of each column
for col in numeric_features:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

#  Final feature set

features = ["text"] + numeric_features
X = df[features].copy()
y = df["target"].copy()

print(f"After cleaning: {df.shape[0]} rows, using features: {features}")
print("Class balance (0=failed, 1=successful):")
print(y.value_counts(normalize=True).round(3))

#  Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocessing & model
# Text vectorizer
text_vectorizer = TfidfVectorizer(
    max_features=20000,          # large enough for signal, not too heavy
    ngram_range=(1, 2),          # unigrams + bigrams
    min_df=2
)

transformers = [
    ("text", text_vectorizer, "text")
]

# Only adding numeric transformer 
if numeric_features:
    transformers.append(("num", StandardScaler(with_mean=False), numeric_features))
    

preprocessor = ColumnTransformer(transformers)

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",   # handle class imbalance robustly
    n_jobs=None                # keep default for compatibility
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", clf)
])

# Light hyperparameter search for a boost without being too slow
param_grid = {
    "clf__C": [0.5, 1.0, 2.0],
    "clf__solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

#  Evaluate

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

#probability based output

success_probs = best_model.predict_proba(X_test)[:, 1] * 100
df_results = pd.DataFrame({
    "success_probability (%)": success_probs.round(2),
    "actual": y_test.values,
    "predicted": y_pred
})

# saving model and accuracy 

joblib.dump(best_model, "pitchmodel.pkl")
print("\nSaved: pitchmodel.pkl")

# --- Function for UI prediction ---
def predict_success_probability(user_input: dict) -> float:
    """
    Takes a dictionary of user inputs and returns success probability (%).
    Example input: {"goal": 5000, "backers": 100, "usd_pledged": 4500, "text": "A smart watch for kids"}
    """
    # Load model
    model = joblib.load("pitchmodel.pkl")

    # Create DataFrame from input
    input_df = pd.DataFrame([user_input])

    # Clean text
    input_df["text"] = input_df["text"].apply(clean_text_basic)

    # Ensure numeric columns are present
    for col in numeric_features:
        if col not in input_df.columns:
            input_df[col] = 0  # default fallback

    # Predict probability
    prob = model.predict_proba(input_df)[0, 1] * 100  # success class
    return round(prob, 2)

def explain_prediction(user_input: dict):
    """
    Returns SHAP values and a summary plot for a single prediction.
    """
    # Load model
    model = joblib.load("pitchmodel.pkl")

    # Create DataFrame from input
    input_df = pd.DataFrame([user_input])
    input_df["text"] = input_df["text"].apply(clean_text_basic)

    # Fill missing numeric features
    for col in numeric_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Preprocess input
    transformed_input = model.named_steps["preprocess"].transform(input_df)

    # Create SHAP explainer
    explainer = shap.Explainer(model.named_steps["clf"], transformed_input)
    shap_values = explainer(transformed_input)

    return shap_values

def get_shap_values(user_input: dict):
    import shap
    model = joblib.load("pitchmodel.pkl")

    input_df = pd.DataFrame([user_input])
    input_df["text"] = input_df["text"].apply(clean_text_basic)

    for col in numeric_features:
        if col not in input_df.columns:
            input_df[col] = 0

    transformed_input = model.named_steps["preprocess"].transform(input_df)
    explainer = shap.Explainer(model.named_steps["clf"], transformed_input)
    shap_values = explainer(transformed_input)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    return shap_values, feature_names

# In FoundersFuelFrontend/vc-copilot-backend/trainmodel.py (REPLACE existing function)

def generate_narrative_summary(user_input: dict, shap_values, feature_names, probability: float) -> str:
    impact = shap_values.values[0]
    
    # START LIST
    summary_list = [
        f"<li>🔍 Estimated success probability: **{probability:.2f}%**.</li>"
    ]

    feature_impacts = dict(zip(feature_names, impact))
    text_comment_added = False

    # Always check key numeric features
    for key in ["goal", "backers", "usd_pledged"]:
        # Find the feature name corresponding to the key (handles case like 'num__goal')
        feature = next((f for f in feature_names if key in f), None)
        
        if feature:
            value = feature_impacts[feature]
            direction = "boosted" if value > 0 else "lowered"
            emoji = "✅" if value > 0 else "⚠️"
            val = user_input.get(key, 0)
            
            phrase = ""
            if key == "goal":
                phrase = f"{emoji} **Funding Goal:** Your goal of ${val:,.0f} {direction} the prediction. Moderate goals signal feasibility, while very high ones may deter backers."
            elif key == "backers":
                phrase = f"{emoji} **Backer Momentum:** Having {val} backers {direction} the success likelihood. Strong early support is a key signal."
            elif key == "usd_pledged":
                phrase = f"{emoji} **Pledged Amount:** A pledged amount of ${val:,.0f} {direction} the outcome. Higher pledges suggest momentum."

            if phrase:
                summary_list.append(f"<li>{phrase}</li>")

    # Then check top SHAP features for text
    for feature, value in sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True):
        if "text" in feature and not text_comment_added:
            direction = "boosted" if value > 0 else "lowered"
            emoji = "✅" if value > 0 else "⚠️"
            
            # Using the original feature name here might show things like 'text__smart watch'
            # We simplify for the user facing narrative.
            
            phrase = f"{emoji} **Pitch Quality:** The core wording of your pitch {direction} the prediction. Clear, engaging language performs better."
            summary_list.append(f"<li>{phrase}</li>")
            text_comment_added = True
            break # Only need one general text comment

    # Concluding advice
    if probability < 30:
        advice = "🚨 **Recommendation:** This pitch may struggle to gain traction. Consider revising the language to be more specific, emotionally engaging, and clearly targeted. Avoid generic phrases and highlight what makes your project unique."
    else:
        advice = "💡 **Next Steps:** To improve your chances, consider refining your pitch to emphasize urgency, uniqueness, and social proof. Highlight what makes your project stand out and why it matters now."
        
    summary_list.append(f"<li>{advice}</li>")
    
    # Join all list items and wrap in <ul> and an introductory heading
    return "<h4>Detailed Analysis:</h4><ul>" + "".join(summary_list) + "</ul>"

def get_shap_explanation(user_input: dict):
    
    model = joblib.load("pitchmodel.pkl")
    input_df = pd.DataFrame([user_input])
    input_df["text"] = input_df["text"].apply(clean_text_basic)

    for col in numeric_features:
        if col not in input_df.columns:
            input_df[col] = 0

    transformed_input = model.named_steps["preprocess"].transform(input_df)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.Explainer(model.named_steps["clf"], transformed_input)
    shap_values = explainer(transformed_input)
    probability = model.predict_proba(input_df)[0, 1] * 100

    return shap_values, feature_names, probability