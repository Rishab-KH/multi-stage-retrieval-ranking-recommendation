from src.data_processing import *
import os

data_dir = 'data'

# Step 1: Load data
orders, interactions, order_products = load_and_merge_data(data_dir)
print(f"Orders shape: {orders.shape}")
print(f"Interactions shape: {interactions.shape}")
print(f"Order products shape: {order_products.shape}")

# Step 2: Filter active users
interactions = filter_active_users(orders, interactions, min_orders=3)
print(f"After filtering: {interactions.shape}")

# Step 3: Temporal split
train, test = temporal_train_test_split(interactions)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Step 4: Build mappings
user2idx, prod2idx = build_mappings(train, test)
print(f"Users: {len(user2idx)}, Products: {len(prod2idx)}")

# Step 5: Map to indices
train_indexed = interactions_to_indices(train, user2idx, prod2idx)
test_indexed = interactions_to_indices(test, user2idx, prod2idx)
print(f"Train indexed shape: {train_indexed.shape}")
print(f"Test indexed shape: {test_indexed.shape}")