# NBA Game Outcome Predictor

[Project Link: https://github.com/sahilchoudhari08/NBA-Game-Outcome-Predictor](https://github.com/sahilchoudhari08/NBA-Game-Outcome-Predictor)

This project is an interactive web app that predicts the outcome of NBA games using the latest team and player statistics, historical data, and machine learning. The app features a clean, table-based UI for selecting teams, viewing predictions, and analyzing recent form and head-to-head history.

## Features
- **Predict NBA game outcomes** using up-to-date stats and a trained machine learning model
- **Team selection**: Choose any two NBA teams for a matchup
- **Prediction results**: See the predicted winner and win probability
- **Recent form tables**: View each team's last 5 games and results
- **Head-to-head table**: See the last 5 games between the selected teams, with scores and outcomes
- **No confusing graphs**: All analysis is presented in easy-to-read tables

## How It Works
- **Data Source**: Live and historical NBA data is fetched using the [nba_api](https://github.com/swar/nba_api) Python package
- **Feature Engineering**: The app computes recent and season-long stats for both teams, including win percentage, net rating, and more
- **Model**: An XGBoost classifier is trained on historical NBA games, using the same features as the prediction pipeline
- **UI**: Built with [Streamlit](https://streamlit.io/), the app provides a simple, modern interface focused on actionable tables

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sahilchoudhari08/NBA-Game-Outcome-Predictor.git
   cd NBA-Game-Outcome-Predictor
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the model** (required before first use)
   ```bash
   python train_model.py
   ```
5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
.
├── app.py                # Main Streamlit app
├── train_model.py        # Script to train the model
├── src/
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── visualization.py
├── models/               # Saved model and scaler
├── data/                 # Cached data
├── requirements.txt
└── README.md
```

## Contributing
Pull requests are welcome! If you have ideas for new features, bug fixes, or improvements, please open an issue or submit a PR.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Disclaimer:** This app is for educational and informational purposes only. Predictions are based on historical data and statistical analysis, and should not be used as the sole basis for betting or decision-making. 
