Smart Travel & Product Insight Platform - README
=================================================

This Flask-based project offers multiple intelligent tools:

- Product price forecasting
- Hotel price prediction
- Flight price and cancellation prediction
- Customer comment sentiment analysis
- Product search chatbot
- Product recommendation system


HOW TO SET UP
-------------

1. Requirements:
   - Python 3.8+
   - MySQL Server (running locally)
   - pip packages from requirements.txt

2. Database Setup:
   - Create a MySQL database named: product_catalogue
   - Run:
     - insertfileintophp.py (loads product CSV data)
     - insert_hoteltinto_php.py (loads hotel CSV data)

     *IMPORTANT: Update file paths inside the scripts before running*

3. Install dependencies:
   > pip install -r requirements.txt

4. Start the app:
   > python app.py

   Then open your browser to: http://localhost:5000


KEY FEATURES
------------

- /products: Product price forecasting (using Prophet)
- /hotel1: Predicts hotel price per night
- /flight_price: Predicts flight price
- /flight_cancel: Predicts flight cancellation likelihood
- /sentiment: Analyzes comment sentiment (positive/negative/neutral)
- /recommend: Suggests products based on past selections
- /chatbot: Finds products using smart keyword search


FILES AND MODELS
----------------

- model_flight.pkl → Flight price model
- hotel_price_model.pkl → Hotel price model
- sentiment_model.pkl + label_encoder.pkl → Sentiment analysis
- recommender_model.pth → PyTorch recommendation model
- templates/ → HTML templates for each feature
- static/ → Images, CSS and JS
- insertfileintophp.py → Loads product CSV into database
- insert_hoteltinto_php.py → Loads hotel CSV into database


NOTES
-----

- MySQL must be running
- Models are already trained and loaded
- For chatbot to work, make sure data/fullelectronique.csv is present
- Tested on localhost:5000


Author: Groupe Comparateur Prix
