#dont forget to install the connector in the terminal with this command  pip install mysql-connector-python

import pandas as pd
import mysql.connector

# 📂 Step 1: Load your CSV
file_path = r"C:\Users\Adem\Desktop\adem\final_catalogue_all_products_v1_temp_serie.csv"
df = pd.read_csv(file_path)
print(f"✅ Loaded {len(df)} rows from CSV")

# 🛠️ Replace NaN with None (to avoid SQL errors)
df = df.where(pd.notnull(df), None)

# 🌐 Step 2: Connect to MySQL (XAMPP default config)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="adem",  # default in XAMPP
    database="product_catalogue"
)
cursor = conn.cursor()

# 🧱 Step 3: Create the database if not exists
cursor.execute("CREATE DATABASE IF NOT EXISTS product_catalogue")
cursor.execute("USE product_catalogue")

# 🧱 Step 4: Create the table (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS all_products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_full TEXT,
    marque VARCHAR(255),
    available VARCHAR(50),
    store VARCHAR(100),
    category VARCHAR(100),
    prix_2025_winter FLOAT,
    prix_2024_fall FLOAT,
    prix_2024_summer FLOAT,
    prix_2024_spring FLOAT,
    prix_2024_winter FLOAT,
    prix_2023_fall FLOAT,
    prix_2023_summer FLOAT,
    prix_2023_spring FLOAT,
    prix_2023_winter FLOAT,
    prix_2022_fall FLOAT,
    prix_2022_summer FLOAT,
    prix_2022_spring FLOAT,
    prix_2022_winter FLOAT,
    prix_2021_fall FLOAT,
    prix_2021_summer FLOAT,
    prix_2021_spring FLOAT,
    prix_2021_winter FLOAT,
    prix_2020_fall FLOAT,
    prix_2020_summer FLOAT,
    prix_2020_spring FLOAT,
    prix_2020_winter FLOAT,
    prix_2019_fall FLOAT,
    prix_2019_summer FLOAT,
    prix_2019_spring FLOAT,
    prix_2019_winter FLOAT,
    prix_2018_fall FLOAT,
    prix_2018_summer FLOAT,
    prix_2018_spring FLOAT,
    prix_2018_winter FLOAT
)
""")
print("✅ Table ready")

# 📦 Step 5: Prepare Insert data as tuples
columns_to_insert = [
    "product_full", "marque", "available", "store", "category",
    "prix_2025_winter", "prix_2024_fall", "prix_2024_summer", "prix_2024_spring", "prix_2024_winter",
    "prix_2023_fall", "prix_2023_summer", "prix_2023_spring", "prix_2023_winter",
    "prix_2022_fall", "prix_2022_summer", "prix_2022_spring", "prix_2022_winter",
    "prix_2021_fall", "prix_2021_summer", "prix_2021_spring", "prix_2021_winter",
    "prix_2020_fall", "prix_2020_summer", "prix_2020_spring", "prix_2020_winter",
    "prix_2019_fall", "prix_2019_summer", "prix_2019_spring", "prix_2019_winter",
    "prix_2018_fall", "prix_2018_summer", "prix_2018_spring", "prix_2018_winter"
]
data = df[columns_to_insert].values.tolist()

# 🧩 Step 6: Safe insert (100 rows at a time)
insert_query = f"""
INSERT INTO all_products ({', '.join(columns_to_insert)})
VALUES ({', '.join(['%s'] * len(columns_to_insert))})
"""

batch_size = 100
for i in range(0, len(data), batch_size):
    try:
        batch = data[i:i + batch_size]
        cursor.executemany(insert_query, batch)
        conn.commit()
        print(f"✅ Inserted rows {i} to {i + len(batch)}")
    except Exception as e:
        print(f"❌ Failed to insert rows {i} to {i + batch_size}: {e}")
        break

# ✅ Step 7: Clean up
cursor.close()
conn.close()
print("🎉 All data inserted successfully!")
