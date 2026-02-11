# app.py
# Flask web app for flight price prediction using a trained KNN model
# Feature order MUST match training:
# [airline, source_city, destination_city, class, stops,
#  departure_time, arrival_time, duration, days_left]

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        airline = float(request.form['airline'])
        source = float(request.form['source'])
        destination = float(request.form['destination'])
        flight_class = float(request.form['flight_class'])
        stops = float(request.form['stops'])
        departure_time = float(request.form['departure_time'])
        arrival_time = float(request.form['arrival_time'])
        duration = float(request.form['duration'])
        days_left = float(request.form['days_left'])

        X = np.array([[
            airline,
            source,
            destination,
            flight_class,
            stops,
            departure_time,
            arrival_time,
            duration,
            days_left
        ]])

        X_scaled = scaler.transform(X)
        prediction = round(knn_model.predict(X_scaled)[0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
