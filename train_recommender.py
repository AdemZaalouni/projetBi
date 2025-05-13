import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mysql.connector
import matplotlib.pyplot as plt

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="product_catalogue"
)
'''''
# 🔄 Load Data
print("🔄 Loading dim_client...")
dim_client = pd.read_sql("SELECT client_id, produit_achete FROM dim_client", conn)

print("🔄 Loading all_products...")
all_products = pd.read_sql("SELECT id, product_full, marque, category, prix_2025_winter FROM all_products", conn)
conn.close()

# 🛠 Prepare all_products
all_products['product_full'] = all_products['product_full'].str.strip()
all_products['marque'] = all_products['marque'].fillna("Unknown")
all_products['category'] = all_products['category'].fillna("Unknown")

# 🔀 Merge
print("🔄 Merging client purchases with products...")
enhanced = dim_client.merge(
    all_products,
    left_on="produit_achete",
    right_on="product_full",
    how="left"
)

# 🛠 Handle missing matches silently
missing = enhanced[enhanced['id'].isna()]
if not missing.empty:
    for idx in tqdm(missing.index, desc="Completing product info"):
        random_product = all_products.sample(1).iloc[0]
        enhanced.at[idx, 'id'] = random_product['id']
        enhanced.at[idx, 'product_full'] = random_product['product_full']
        enhanced.at[idx, 'marque'] = random_product['marque']
        enhanced.at[idx, 'category'] = random_product['category']
        enhanced.at[idx, 'prix_2025_winter'] = random_product['prix_2025_winter']

# 🛠 Final formatting
enhanced['categorie_enc'] = pd.factorize(enhanced['category'])[0]
enhanced['produit_enc'] = pd.factorize(enhanced['product_full'])[0]
enhanced['prix_2025_winter'] = enhanced['prix_2025_winter'].fillna(0)
enhanced['rating'] = np.log1p(enhanced['prix_2025_winter'])

final = enhanced[['client_id', 'id', 'product_full', 'marque', 'category', 'prix_2025_winter', 'categorie_enc', 'produit_enc', 'rating']].rename(columns={
    'id': 'product_id',
    'product_full': 'product_name',
    'marque': 'brand',
    'category': 'category',
    'prix_2025_winter': 'price'
})

# 💾 Save enhanced data
final.to_csv("client_product_enhanced_final.csv", index=False)
print("✅ Saved: client_product_enhanced_final.csv")
'''
# ============================================
# 🔥 Train Recommender
# ============================================
df = pd.read_csv("client_product_enhanced_final.csv")

df['user_enc'], user_index = pd.factorize(df['client_id'])
df['item_enc'], item_index = pd.factorize(df['product_id'])
n_users = df['user_enc'].nunique()
n_items = df['item_enc'].nunique()
n_brands = df['brand'].nunique()
n_categories = df['categorie_enc'].nunique()

class PurchaseDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_enc'].values, dtype=torch.long)
        self.items = torch.tensor(dataframe['item_enc'].values, dtype=torch.long)
        self.brands = torch.tensor(dataframe['brand'].astype('category').cat.codes.values, dtype=torch.long)
        self.categories = torch.tensor(dataframe['categorie_enc'].values, dtype=torch.long)
        self.prices = torch.tensor(dataframe['price'].values, dtype=torch.float32)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.brands[idx], self.categories[idx], self.prices[idx], self.ratings[idx]

class RecommenderNN(nn.Module):
    def __init__(self, n_users, n_items, n_brands, n_categories, emb_size=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.brand_emb = nn.Embedding(n_brands, emb_size // 2)
        self.cat_emb = nn.Embedding(n_categories, emb_size // 2)
        self.fc = nn.Sequential(
            nn.Linear(emb_size * 2 + emb_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, item, brand, category, price):
        u = self.user_emb(user)
        i = self.item_emb(item)
        b = self.brand_emb(brand)
        c = self.cat_emb(category)
        x = torch.cat([u, i, b, c, price.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(1)

# Train/val split
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = PurchaseDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderNN(n_users, n_items, n_brands, n_categories).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
epochs = 5
train_losses = []
print("🚀 Training model...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for user, item, brand, category, price, rating in train_loader:
        user, item, brand, category, price, rating = user.to(device), item.to(device), brand.to(device), category.to(device), price.to(device), rating.to(device)
        optimizer.zero_grad()
        preds = model(user, item, brand, category, price)
        loss = loss_fn(preds, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Save model and mappings
torch.save(model.state_dict(), "recommender_model.pth")
user_index.to_series().to_csv("user_mapping.csv")
item_index.to_series().to_csv("item_mapping.csv")
print("✅ Model + mappings saved")

# Plot training curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
