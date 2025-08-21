import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('adult 3.csv')

# Drop the 'fnlwgt' column as it's not useful for prediction
df = df.drop(['fnlwgt'], axis=1)

# Handle missing values represented by '?'
for column in df.columns:
    if '?' in df[column].unique():
        df[column] = df[column].replace('?', df[column].mode()[0])

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model and label encoder
joblib.dump(model, 'salary_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

print("Model trained and saved successfully!")