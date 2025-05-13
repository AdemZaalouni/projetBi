import pandas as pd
import mysql.connector

# 📂 Step 1: Load the hotel CSV
file_path = r"C:\Users\Adem\Desktop\adem\final_hotels_seasonal_price_serie.csv"
df = pd.read_csv(file_path)
print(f"✅ Loaded {len(df)} rows from CSV")

# 🛠 Step 2: Connect to MySQL (XAMPP defaults)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="adem",  # Default XAMPP: empty password
    database="product_catalogue"
)
cursor = conn.cursor()

# 🧱 Step 3: Create the table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS hotels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name TEXT,
    price FLOAT,
    formule VARCHAR(255),
    nb_etoiles INT,
    rating FLOAT,
    city VARCHAR(100),
    distance_centre_ville FLOAT,
    agency_name VARCHAR(100),
    nbr_travellers INT,
    creation_date VARCHAR(100),
    prix_2025 FLOAT,
    prix_2024_winter FLOAT,
    prix_2024_fall FLOAT,
    prix_2024_summer FLOAT,
    prix_2024_spring FLOAT,
    prix_2023_winter FLOAT,
    prix_2023_fall FLOAT,
    prix_2023_summer FLOAT,
    prix_2023_spring FLOAT,
    prix_2022_winter FLOAT,
    prix_2022_fall FLOAT,
    prix_2022_summer FLOAT,
    prix_2022_spring FLOAT,
    prix_2021_winter FLOAT,
    prix_2021_fall FLOAT,
    prix_2021_summer FLOAT,
    prix_2021_spring FLOAT,
    prix_2020_winter FLOAT,
    prix_2020_fall FLOAT,
    prix_2020_summer FLOAT,
    prix_2020_spring FLOAT,
    prix_2019_winter FLOAT,
    prix_2019_fall FLOAT,
    prix_2019_summer FLOAT,
    prix_2019_spring FLOAT,
    prix_2018_winter FLOAT,
    prix_2018_fall FLOAT,
    prix_2018_summer FLOAT,
    prix_2018_spring FLOAT
)
""")
print("✅ Table 'hotels' ready")

# 🧱 Step 4: Insert data in chunks
columns = [
    "name", "price", "formule", "nb_etoiles", "rating", "city", "distance_centre_ville",
    "agency_name", "nbr_travellers", "creation_date",
    "prix_2025", "prix_2024_winter", "prix_2024_fall", "prix_2024_summer", "prix_2024_spring",
    "prix_2023_winter", "prix_2023_fall", "prix_2023_summer", "prix_2023_spring",
    "prix_2022_winter", "prix_2022_fall", "prix_2022_summer", "prix_2022_spring",
    "prix_2021_winter", "prix_2021_fall", "prix_2021_summer", "prix_2021_spring",
    "prix_2020_winter", "prix_2020_fall", "prix_2020_summer", "prix_2020_spring",
    "prix_2019_winter", "prix_2019_fall", "prix_2019_summer", "prix_2019_spring",
    "prix_2018_winter", "prix_2018_fall", "prix_2018_summer", "prix_2018_spring"
]

data = df[columns].values.tolist()

# Insert in batches
insert_query = f"""
INSERT INTO hotels ({', '.join(columns)})
VALUES ({', '.join(['%s'] * len(columns))})
"""

batch_size = 100
for i in range(0, len(data), batch_size):
    try:
        batch = data[i:i+batch_size]
        cursor.executemany(insert_query, batch)
        conn.commit()
        print(f"✅ Inserted rows {i} to {i+len(batch)}")
    except Exception as e:
        print(f"❌ Error inserting rows {i}-{i+batch_size}: {e}")
        break

# ✅ Wrap it up
cursor.close()
conn.close()
print("🎉 Hotel data inserted successfully!")
