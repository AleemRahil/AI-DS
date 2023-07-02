import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load the survey data into a DataFrame
survey_data = pd.read_csv('survey_data.csv')

# Preprocess the data (e.g., handle missing values, encode categorical variables)

# Split the data into training and testing sets
X = survey_data.drop('sentiment', axis=1)  # Features (excluding the target variable)
y = survey_data['sentiment']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions using logistic regression
lr_predictions = lr_model.predict(X_test)

# Evaluate logistic regression model's performance
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)

# Train decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Make predictions using decision tree
dt_predictions = dt_model.predict(X_test)

# Evaluate decision tree model's performance
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)

# Train XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost
xgb_predictions = xgb_model.predict(X_test)

# Evaluate XGBoost model's performance
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("XGBoost Accuracy:", xgb_accuracy)
