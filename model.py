import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path
from feature_engineering import FeatureEngineer
from data_collection import NBADataCollector

class NBAPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self.engineer = FeatureEngineer()
        self.collector = NBADataCollector()
        
    def prepare_training_data(self, historical_games):
        """Prepare historical games data for training using the same features as prediction"""
        # Add OPPONENT_TEAM_ID if missing
        if 'OPPONENT_TEAM_ID' not in historical_games.columns:
            def get_opponent_team_id(row):
                if 'vs.' in row['MATCHUP']:
                    opp_abbr = row['MATCHUP'].split('vs.')[1].strip()
                elif '@' in row['MATCHUP']:
                    opp_abbr = row['MATCHUP'].split('@')[1].strip()
                else:
                    return None
                for team in self.collector.teams:
                    if team['abbreviation'] == opp_abbr:
                        return team['id']
                return None
            historical_games['OPPONENT_TEAM_ID'] = historical_games.apply(get_opponent_team_id, axis=1)
        features_list = []
        labels = []
        # Sort games by date ascending
        historical_games = historical_games.sort_values('GAME_DATE')
        # For each game, compute features for both teams using games prior to this game
        for idx, row in historical_games.iterrows():
            home_team_id = row['TEAM_ID']
            away_team_id = row['OPPONENT_TEAM_ID']
            game_date = row['GAME_DATE']
            # Get all games for each team before this game
            home_team_games = historical_games[(historical_games['TEAM_ID'] == home_team_id) & (historical_games['GAME_DATE'] < game_date)]
            away_team_games = historical_games[(historical_games['TEAM_ID'] == away_team_id) & (historical_games['GAME_DATE'] < game_date)]
            # Skip if not enough games
            if len(home_team_games) < 5 or len(away_team_games) < 5:
                continue
            # Compute features
            features = self.engineer.prepare_features(home_team_games.tail(10), away_team_games.tail(10))
            features_list.append(features.iloc[0])
            # Label: 1 if home team won, 0 otherwise
            labels.append(1 if row['WL'] == 'W' else 0)
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        return X, y
        
    def train(self, X, y):
        """Train the XGBoost model with hyperparameter tuning"""
        # Save feature names
        self.feature_names = X.columns.tolist()
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # Define parameter grid for tuning (simpler, faster)
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        # Initialize base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        # Get best model
        self.model = grid_search.best_estimator_
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        # Save model, scaler, and feature names
        self.save_model()
        self.save_feature_names()
        
    def predict(self, features):
        """Make prediction for a single game"""
        if self.model is None:
            self.load_model()
        self.load_feature_names()
        # Reindex features to match training
        features = features.reindex(columns=self.feature_names, fill_value=0)
        print("Features used for prediction:")
        print(features)
        # Warn if all features are zero or identical
        if (features.nunique().sum() == 0) or (features.sum(axis=1).iloc[0] == 0):
            print("WARNING: All features are zero or identical. Model cannot make a meaningful prediction.")
            warning = "Warning: All features are zero or identical. Model cannot make a meaningful prediction."
        else:
            warning = None
        # Scale features
        features_scaled = self.scaler.transform(features)
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        return {
            'prediction': prediction,
            'probability': probability[1] if prediction == 1 else probability[0],
            'feature_importance': self.get_feature_importance(features.columns),
            'warning': warning
        }
        
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))
        
    def save_model(self):
        """Save model and scaler to disk"""
        model_path = self.model_dir / 'nba_predictor.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
    def load_model(self):
        """Load model and scaler from disk"""
        model_path = self.model_dir / 'nba_predictor.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model files not found. Please train the model first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def save_feature_names(self):
        feature_path = self.model_dir / 'feature_names.txt'
        with open(feature_path, 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
    def load_feature_names(self):
        feature_path = self.model_dir / 'feature_names.txt'
        if not feature_path.exists():
            raise FileNotFoundError("Feature names file not found. Please train the model first.")
        with open(feature_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()] 