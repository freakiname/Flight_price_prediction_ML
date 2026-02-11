# Flight Price Prediction (Prédiction Prix Vols)

A machine learning project that predicts flight prices using a K-Nearest Neighbors (KNN) regressor. The model is trained on Indian domestic flight data and outputs prices in Moroccan Dirhams (DH). A Flask web application provides an interactive interface for real-time predictions.

## Overview

This project combines data mining techniques with a lightweight web interface to help users estimate flight prices based on various factors such as airline, route, class, number of stops, and booking timing.

**Key features:**
- KNN-based regression model with high accuracy (R² ≈ 0.976)
- StandardScaler for feature normalization
- Web interface with intuitive form inputs
- Price output in Moroccan Dirhams (DH)

## Project Structure

```
Prediction_prix_vols/
├── app.py              # Flask web application
├── Train_model.ipynb   # Jupyter notebook: data processing, training, model export
├── knn_model.pkl       # Serialized KNN model
├── scaler.pkl          # Serialized StandardScaler (for feature scaling)
├── templates/
│   └── index.html      # Web form for flight price prediction
└── README.md           # Project documentation
```

## Dataset

The model is trained on the **Flight Price Prediction** dataset from Kaggle:
- **Source:** [shubhambathwal/flight-price-prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- **File:** `Clean_dataset.csv`
- **Download:** Via `kagglehub` in the notebook

**Features:**
| Feature          | Description                          |
|------------------|--------------------------------------|
| airline          | Airline carrier                      |
| source_city      | Departure city                       |
| destination_city | Arrival city                         |
| class            | Economy or Business                  |
| stops            | Number of stops (zero, one, two+)    |
| departure_time   | Time of departure                    |
| arrival_time     | Time of arrival                      |
| duration         | Flight duration (hours)              |
| days_left        | Days until departure                 |

**Target:** Price (converted from Indian Rupees to Moroccan Dirhams using 0.1037 rate)

## Model

- **Algorithm:** K-Nearest Neighbors Regressor
- **Parameters:** `n_neighbors=5`, `metric='manhattan'`
- **Preprocessing:** Label encoding for categorical features, Ordinal encoding for ordered categories, StandardScaler for normalization
- **Performance (test set):**
  - MAE: ~174 DH
  - RMSE: ~364 DH
  - R²: ~0.976

## Installation

### Prerequisites

- Python 3.8+
- Jupyter (for running the training notebook)

### Dependencies

```bash
pip install flask numpy scikit-learn joblib pandas kagglehub matplotlib seaborn
```

For a minimal setup to run only the web app:

```bash
pip install flask numpy scikit-learn joblib
```

## Usage

### 1. Run the Web Application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 2. Retrain the Model

1. Open `Train_model.ipynb` in Jupyter.
2. Run all cells to:
   - Download the dataset via kagglehub
   - Load and clean the data
   - Convert prices to Moroccan Dirhams
   - Encode categorical features
   - Train the KNN model
   - Save `knn_model.pkl` and `scaler.pkl`

**Note:** Ensure the feature order in `app.py` matches the training pipeline if you modify the notebook.

### 3. Form Input Reference

| Field        | Options / Format                                           |
|-------------|-------------------------------------------------------------|
| Airline     | IndiGo (0), Air India (1), Vistara (2), SpiceJet (3), GO_FIRST (4) |
| Source City | Delhi (0), Mumbai (1), Bangalore (2), Kolkata (3), Hyderabad (4), Chennai (5) |
| Destination | Same as Source City                                        |
| Departure Time | Early Morning (0), Morning (1), Afternoon (2), Evening (3), Night (4) |
| Arrival Time   | Same as Departure Time                                  |
| Flight Class | Economy (0), Business (1)                                  |
| Stops       | Non-stop (0), 1 Stop (1), 2+ Stops (2)                     |
| Duration    | Flight duration in **hours** (e.g. 2.17 for 2h 10min)      |
| Days Left   | Number of days until departure                             |

## Technologies

- **Backend:** Flask, scikit-learn
- **Model:** KNeighborsRegressor
- **Data:** pandas, kagglehub
- **Visualization (notebook):** matplotlib, seaborn

## License

This project is for educational purposes created by Mohamed Amine Rezoum.
