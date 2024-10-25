# Predict for a specific user and product
user_id = 1
product_id = 104  # New product to predict
pred = model.predict(user_id, product_id)
print(f"Prediction for User {user_id} on Product {product_id}: {pred.est}")
