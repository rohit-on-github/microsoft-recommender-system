import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample dataset (users, products, ratings)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'product_id': [101, 102, 103, 101, 103, 101, 104, 102, 104, 103],
    'rating': [5, 3, 4, 2, 4, 5, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Build Collaborative Filtering Model (User-Based)
sim_options = {
    'name': 'cosine',
    'user_based': True  # User-based collaborative filtering
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Predicting and evaluating the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Predict for a specific user and product
user_id = 1
product_id = 104  # New product to predict
pred = model.predict(user_id, product_id)
print(f"Prediction for User {user_id} on Product {product_id}: {pred.est}")

# Get top-N product recommendations for a user
def get_top_n_recommendations(predictions, n=5):
    top_n = {}
    
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    # Sort predictions for each user and get the top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get recommendations for all users in the test set
top_n = get_top_n_recommendations(predictions, n=3)
for uid, user_ratings in top_n.items():
    print(f"User {uid}'s top recommendations: {user_ratings}")
