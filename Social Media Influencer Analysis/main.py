import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Data Collection

# Use Instagram API to collect user profiles and engagement metrics
def get_user_profile(username):
    access_token = "<YOUR_INSTAGRAM_ACCESS_TOKEN>"
    url = f"https://api.instagram.com/v1/users/{username}/?access_token={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        user_data = response.json()
        user_id = user_data['data']['id']
        follower_count = user_data['data']['counts']['followed_by']
        post_count = user_data['data']['counts']['media']
        return user_id, follower_count, post_count
    else:
        return None, None, None

# Example user profiles
usernames = ["user1", "user2", "user3"]
data = []

for username in usernames:
    user_id, follower_count, post_count = get_user_profile(username)
    if user_id:
        data.append({
            "username": username,
            "follower_count": follower_count,
            "post_count": post_count
        })

# Step 2: Preprocessing
df = pd.DataFrame(data)

# Drop duplicates and irrelevant profiles
df.drop_duplicates(subset="username", keep="first", inplace=True)
df.dropna(subset=["follower_count", "post_count"], inplace=True)

# Step 3: Feature Engineering
# Example: Calculate engagement rate
df["engagement_rate"] = df["post_count"] / df["follower_count"]

# Step 4: Training and Evaluation
# Assuming you have a labeled dataset with influencer/non-influencer labels
X = df[["follower_count", "engagement_rate"]]  # Removing 'post_count' as it may not be a strong predictor
y = df["label"]  # Replace 'label' with your actual target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Increase the number of estimators for better performance
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 5: Recommendations and Visualization
# Assuming you have a new dataset of potential influencers to predict
X_new = pd.DataFrame(...)
predictions = clf.predict(X_new)