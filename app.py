import logging  # ✅ Corrigé : utiliser logging au lieu de venv
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import mysql.connector
import os
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mysql.connector
from flask import jsonify, request
import bcrypt

# Configurer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les modèles et données
model_pipeline = joblib.load('sentiment_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
df_elec = pd.read_csv("data/fullelectronique (2).csv")

# Charger model_flight et features_encoded_flight
model_flight = None
features_encoded_flight = None
try:
    model_path = os.path.join(r"C:\Users\Adem\Desktop\flask_deployment", "model_flight.pkl")
    logger.info(f"Tentative de chargement du modèle depuis : {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier {model_path} n'existe pas")
    model_flight = joblib.load(model_path)
    features_encoded_flight = getattr(model_flight, 'feature_names_in_', None)
    if features_encoded_flight is None:
        logger.warning("feature_names_in_ non disponible. Définir manuellement les colonnes.")
        # Remplacez par les colonnes réelles de votre modèle
        features_encoded_flight = [
            'Nombre_Escales', 'Taxe_Price',
            'AirlineName_AirFrance', 'AirlineName_Lufthansa',  # Exemple
            'Region_Europe', 'Region_Asia',  # Exemple
            'Mois', 'Jour'
        ]
    logger.info(f"Modèle chargé. Colonnes : {features_encoded_flight}")
except Exception as e:
    logger.error(f"Erreur lors du chargement de model_flight : {str(e)}")

model_hotel = joblib.load("hotel_price_model.pkl")
features_hotel = model_hotel.feature_names_in_

app = Flask(__name__)
CORS(app)

os.makedirs("static", exist_ok=True)

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="adem",
        database="product_catalogue"
    )

def load_products():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT product_full FROM all_products", conn)
    conn.close()
    return df['product_full'].dropna().tolist()

PRODUCT_LIST = load_products()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/products', methods=['GET', 'POST'])
def products():
    history_plot = None
    forecast_plot = None
    selected_product = None

    if request.method == 'POST':
        selected_product = request.form['product_name']

        conn = get_connection()
        df = pd.read_sql("SELECT * FROM all_products", conn)
        conn.close()

        price_cols = [col for col in df.columns if col.startswith("prix_")]
        df_long = df.melt(
            id_vars=["product_full", "marque", "category", "available", "store"],
            value_vars=price_cols,
            var_name="periode",
            value_name="prix_saisonnier"
        )
        df_long[['year', 'season']] = df_long['periode'].str.extract(r"prix_(\d{4})_(\w+)?")
        df_long['year'] = df_long['year'].fillna('2025').astype(int)
        df_long['season'] = df_long['season'].fillna('winter')
        season_map = {'winter': 1, 'spring': 4, 'summer': 7, 'fall': 10}
        df_long['month'] = df_long['season'].map(season_map)
        df_long['date'] = pd.to_datetime(df_long[['year', 'month']].assign(day=1))

        subset = df_long[df_long['product_full'] == selected_product][['date', 'prix_saisonnier']].dropna()

        if not subset.empty:
            unique_id = str(uuid.uuid4())[:8]

            plt.figure(figsize=(10, 5))
            plt.plot(subset['date'], subset['prix_saisonnier'], marker='o')
            plt.title(f"Price History: {selected_product}")
            plt.xlabel("Date")
            plt.ylabel("Price (DT)")
            plt.grid(True)
            plt.tight_layout()
            history_plot = f"static/history_{unique_id}.png"
            plt.savefig(history_plot)
            plt.close()

            ts = subset.rename(columns={'date': 'ds', 'prix_saisonnier': 'y'})
            model = Prophet(yearly_seasonality=True)
            model.fit(ts)
            future = model.make_future_dataframe(periods=4, freq='Q')
            forecast = model.predict(future)

            plt.figure(figsize=(10, 5))
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.4)
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            plt.scatter(ts['ds'], ts['y'], color='black', label='Actual')
            plt.title(f"Price Forecast: {selected_product}")
            plt.xlabel("Date")
            plt.ylabel("Forecasted Price (DT)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            forecast_plot = f"static/forecast_{unique_id}.png"
            plt.savefig(forecast_plot)
            plt.close()

    return render_template('products.html',
                           product_list=PRODUCT_LIST,
                           selected_product=selected_product,
                           history_plot=history_plot,
                           forecast_plot=forecast_plot)
def get_connection2():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="adem",
        database="role"
    )
ALLOWED_ROLES = ['Product Manager', 'Hotel Manager', 'Flight Manager']
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'adem',
    'database': 'role'
}
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400

        required_keys = ['firstname', 'lastname', 'email', 'password', 'role']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({'error': f'Clés manquantes : {missing_keys}'}), 400

        firstname = data['firstname']
        lastname = data['lastname']
        email = data['email']
        password = data['password']
        role = data['role']

        # Valider l'email
        if '@' not in email:
            return jsonify({'error': 'Email invalide'}), 400

        # Valider le rôle
        if role not in ALLOWED_ROLES:
            return jsonify({'error': f'Rôle invalide. Rôles autorisés : {ALLOWED_ROLES}'}), 400

        # Hacher le mot de passe
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Connexion à la base de données
        conn = get_connection2()  # Base 'role'
        cursor = conn.cursor()

        # Vérifier si l'email existe déjà
        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'error': 'Cet email est déjà utilisé'}), 400

        # Insérer l'utilisateur
        cursor.execute(
            "INSERT INTO users (firstname, lastname, email, password, role) VALUES (%s, %s, %s, %s, %s)",
            (firstname, lastname, email, hashed_password, role)
        )
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Nouvel utilisateur inscrit : {email} avec le rôle {role}")
        return jsonify({'message': 'Inscription réussie !'}), 201

    except mysql.connector.Error as db_err:
        logger.error(f"Erreur base de données : {str(db_err)}")
        return jsonify({'error': f'Erreur base de données : {str(db_err)}'}), 500
    except Exception as e:
        logger.error(f"Erreur serveur : {str(e)}")
        return jsonify({'error': f'Erreur serveur : {str(e)}'}), 500
@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    connection = get_connection2()
    cursor = connection.cursor()

    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))

    user = cursor.fetchone()

    cursor.close()
    connection.close()

    if user:
        return jsonify({"message": "Connexion réussie", "role": user[4]})  # role est le 5e champ (index 4)
    else:
        return jsonify({"error": "Identifiants incorrects"}), 401


@app.route('/hotels/select_city', methods=['GET', 'POST'])
def select_city():
    conn = get_connection()
    cities = pd.read_sql("SELECT DISTINCT city FROM hotels WHERE city IS NOT NULL", conn)['city'].tolist()
    conn.close()

    if request.method == 'POST':
        selected_city = request.form['city']
        return redirect(url_for('select_hotel', city=selected_city))

    return render_template('select_city.html', cities=cities)

@app.route('/hotels/<city>', methods=['GET', 'POST'])
def select_hotel(city):
    forecast_plot = None
    selected_hotel = None

    conn = get_connection()
    hotels = pd.read_sql("SELECT DISTINCT name FROM hotels WHERE city = %s", conn, params=(city,))
    hotel_list = hotels['name'].dropna().tolist()
    conn.close()

    if request.method == 'POST':
        selected_hotel = request.form['hotel_name']

        conn = get_connection()
        df = pd.read_sql("SELECT * FROM hotels WHERE name = %s", conn, params=(selected_hotel,))
        conn.close()

        price_cols = [col for col in df.columns if col.startswith("prix_")]
        df_long = pd.melt(
            df,
            id_vars=["name", "city", "formule"],
            value_vars=price_cols,
            var_name="periode",
            value_name="prix_saisonnier"
        )
        df_long[['year', 'season']] = df_long['periode'].str.extract(r"prix_(\d{4})_(\w+)?")
        df_long['season'] = df_long['season'].fillna('winter')
        season_map = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4}
        df_long['season_num'] = df_long['season'].map(season_map)
        df_long['date'] = pd.to_datetime(df_long['year'] + '-' + (df_long['season_num'] * 3).astype(str) + '-01')

        subset = df_long[['date', 'prix_saisonnier']].dropna().rename(columns={'date': 'ds', 'prix_saisonnier': 'y'})

        if not subset.empty:
            subset['floor'] = 10
            subset['cap'] = 150

            unique_id = str(uuid.uuid4())[:8]

            model = Prophet(growth='logistic', yearly_seasonality=True)
            model.fit(subset)

            future = model.make_future_dataframe(periods=4, freq='Q')
            future['floor'] = 10
            future['cap'] = 150

            forecast = model.predict(future)

            forecast['yhat'] = forecast['yhat'].clip(lower=10, upper=150)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=10, upper=150)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=10, upper=150)

            plt.figure(figsize=(10, 5))
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.4)
            plt.plot(forecast['ds'], forecast['yhat'], label='Prévision')
            plt.scatter(subset['ds'], subset['y'], color='black', label='Historique')
            plt.title(f"Prévision des prix – {selected_hotel}")
            plt.xlabel("Date")
            plt.ylabel("Prix prévu (DT)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            forecast_plot = f"static/hotel_forecast_{unique_id}.png"
            plt.savefig(forecast_plot)
            plt.close()

    return render_template("select_hotel.html",
                           city=city,
                           hotel_list=hotel_list,
                           selected_hotel=selected_hotel,
                           forecast_plot=forecast_plot)

@app.route('/hotels')
def hotels_redirect():
    return redirect(url_for('select_city'))

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation_form():
    import torch
    import pandas as pd
    from torch import nn

    df = pd.read_csv("client_product_enhanced_final.csv")
    all_products = pd.read_sql("SELECT id, product_full, marque, category, prix_2025_winter FROM all_products", get_connection())
    
    user_index = pd.read_csv("user_mapping.csv", index_col=0, header=None).squeeze()
    item_index = pd.read_csv("item_mapping.csv", index_col=0, header=None).squeeze()

    product_list = [{'id': row['id'], 'name': row['product_full']} for _, row in all_products.iterrows()]
    recommendations = []

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

    df['user_enc'], _ = pd.factorize(df['client_id'])
    df['item_enc'], _ = pd.factorize(df['product_id'])
    n_users = df['user_enc'].nunique()
    n_items = df['item_enc'].nunique()
    n_brands = df['brand'].nunique()
    n_categories = df['categorie_enc'].nunique()

    model = RecommenderNN(n_users, n_items, n_brands, n_categories)
    model.load_state_dict(torch.load("recommender_model.pth", map_location=torch.device('cpu')))
    model.eval()

    if request.method == 'POST':
        selected_ids = request.form.getlist('selected_products')

        if len(selected_ids) >= 5:
            selected_ids = list(map(int, selected_ids))

            candidate_df = df[~df['product_id'].isin(selected_ids)].drop_duplicates('product_id')
            selected_df = df[df['product_id'].isin(selected_ids)]

            brand = torch.tensor(selected_df['brand'].astype('category').cat.codes.mode()[0], dtype=torch.long).repeat(len(candidate_df))
            category = torch.tensor(selected_df['categorie_enc'].mode()[0], dtype=torch.long).repeat(len(candidate_df))
            price = torch.tensor(candidate_df['price'].values, dtype=torch.float32)

            fake_user = torch.zeros(len(candidate_df), dtype=torch.long)
            item = torch.tensor(candidate_df['item_enc'].values, dtype=torch.long)

            with torch.no_grad():
                scores = model(fake_user, item, brand, category, price)

            candidate_df = candidate_df.copy()
            candidate_df['score'] = scores.numpy()
            top_5 = candidate_df.sort_values('score', ascending=False).head(5)

            recommended_ids = top_5['product_id'].tolist()
            recommendations = all_products[all_products['id'].isin(recommended_ids)]['product_full'].tolist()
        else:
            recommendations = ["⚠️ Please select at least 5 products."]

    return render_template('recommend.html',
                           product_list=product_list,
                           recommendations=recommendations)

# Charger model_rf et scaler
with open(r"C:\Users\Adem\Desktop\flask_deployment\model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open(r"C:\Users\Adem\Desktop\flask_deployment\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/flight_cancel', methods=['POST'])
def flight_cancel_predict():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Récupère les données JSON envoyées
            data = request.get_json()

            compagnie = data['Compagnie']
            distance_vol = float(data['Distance_Vol_KM'])
            nombre_escales = int(data['Nombre_Escales'])
            saison_touristique = int(data['Saison_Touristique'])

            # Prépare les caractéristiques pour la prédiction
            features = [[distance_vol, nombre_escales, saison_touristique]]
            X_scaled = scaler.transform(features)

            # Effectue la prédiction avec le modèle
            prediction_val = model_rf.predict(X_scaled)[0]
            prediction = "✈️ Annulé" if prediction_val == 1 else "🛫 Non annulé"

        except Exception as e:
            error = f"Une erreur est survenue : {str(e)}"

    # Retourne la réponse sous forme de JSON
    return jsonify({
        'prediction': prediction,
        'error': error
    })
@app.route('/hotel1', methods=['GET', 'POST'])
def hotel1_home():
    if model_hotel is None:
        return "❌ Modèle hôtel non chargé."

    if request.method == 'POST':
        try:
            input_data = {
                'nb_etoiles': int(request.form['nb_etoiles']),
                'Mois': int(request.form['Mois']),
                'city': request.form['city'],
                'formule': request.form['formule'],
                'name': request.form['name']
            }

            df_input = pd.DataFrame([input_data])
            df_encoded = pd.get_dummies(df_input)

            for col in features_hotel:
                if col not in df_encoded:
                    df_encoded[col] = 0
            df_encoded = df_encoded[features_hotel]

            prediction = model_hotel.predict(df_encoded)[0]
            return render_template("hotels1.html", prediction=round(prediction, 2))

        except Exception as e:
            return render_template("hotels1.html", error=f"Erreur: {str(e)}")

    return render_template("hotels1.html")

@app.route('/api/predict-flight', methods=['POST'])
def api_predict():
    if model_flight is None or features_encoded_flight is None:
        logger.error("Échec de la prédiction : modèle ou colonnes non chargés")
        return jsonify({'error': 'Modèle ou colonnes non chargés'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400

        required_keys = ['Nombre_Escales', 'Taxe_Price', 'AirlineName', 'Region', 'Mois', 'Jour']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({'error': f'Clés manquantes : {missing_keys}'}), 400

        input_data = {
            'Nombre_Escales': int(data['Nombre_Escales']),
            'Taxe_Price': float(data['Taxe_Price']),
            'AirlineName': data['AirlineName'],
            'Region': data['Region'],
            'Mois': int(data['Mois']),
            'Jour': int(data['Jour'])
        }

        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input)

        for col in features_encoded_flight:
            if col not in df_encoded:
                df_encoded[col] = 0
        df_encoded = df_encoded[features_encoded_flight]

        prediction = model_flight.predict(df_encoded)[0]
        logger.info(f"Prédiction réussie : {prediction}")
        return jsonify({'prediction': round(float(prediction), 2)})

    except ValueError as ve:
        logger.error(f"Erreur de type : {str(ve)}")
        return jsonify({'error': f'Erreur de type de données : {str(ve)}'}), 400
    except Exception as e:
        logger.error(f"Erreur serveur : {str(e)}")
        return jsonify({'error': f'Erreur serveur : {str(e)}'}), 500

@app.route('/sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        try:
            commentaire = request.form['commentaire']
            if len(commentaire.strip()) < 5:
                prediction_label = "inconnu"
                message = "Texte trop court ou vide 😶. Veuillez entrer un commentaire valide."
            else:
                input_data = pd.DataFrame([{'commentaire': commentaire}])
                prediction_encoded = model_pipeline.predict(input_data)[0]
                prediction_label = label_encoder.inverse_transform([int(prediction_encoded)])[0]

                if prediction_label == 'positif':
                    message = "Client satisfait 😊🎉!"
                elif prediction_label == 'négatif':
                    message = "Client non satisfait 😞 "
                else:
                    message = "Merci pour votre retour neutre 🙂"

            return render_template('sentiment.html', prediction=prediction_label, message=message)

        except Exception as e:
            return render_template('sentiment.html', error=f"Erreur de prédiction : {str(e)}")
    
    return render_template('sentiment.html')

@app.route('/chatbot')
def chatbot_home():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def chatbot_ask():
    user_input = request.json.get("message")
    corpus = df_elec['product_full'].astype(str).tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)
    query_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    index = similarity.argmax()

    if similarity[index] < 0.2:
        return jsonify({"response": "❌ Désolé, je n'ai pas trouvé ce produit."})

    produit = df_elec.iloc[index]
    response = f"""
💻 {produit['product_full']}
🏷️ Marque: {produit['marque']}
🏬 Vendeur: {produit['Source']}
💵 Prix: {produit['prix']} TND
"""
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)