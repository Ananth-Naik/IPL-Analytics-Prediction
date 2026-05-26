import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
matches = pd.read_csv("dataset/matches.csv")

# Remove rows with missing values
matches = matches.dropna(
    subset=[
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'venue',
        'winner'
    ]
)

# Keep important columns
matches = matches[
    [
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'venue',
        'winner'
    ]
]

# Keep only valid matches
matches = matches[
    (matches['winner'] == matches['team1']) |
    (matches['winner'] == matches['team2'])
]

# Create target column
matches['team1_win'] = (
    matches['winner'] == matches['team1']
).astype(int)

# Save encoders
encoders = {}

categorical_columns = [
    'team1',
    'team2',
    'toss_winner',
    'toss_decision',
    'venue'
]

# Encode categorical columns
for column in categorical_columns:

    encoder = LabelEncoder()

    matches[column] = encoder.fit_transform(
        matches[column]
    )

    encoders[column] = encoder

# Features
X = matches[
    [
        'team1',
        'team2',
        'toss_winner',
        'toss_decision',
        'venue'
    ]
]

# Target
y = matches['team1_win']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(
    model,
    open('model/ipl_model.pkl', 'wb')
)

# Save encoders
pickle.dump(
    encoders,
    open('model/encoders.pkl', 'wb')
)

print("Model Saved Successfully!")
print("Encoders Saved Successfully!")