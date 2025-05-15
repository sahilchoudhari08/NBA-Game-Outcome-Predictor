import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from data_collection import NBADataCollector
from feature_engineering import FeatureEngineer
from model import NBAPredictor

def main():
    print("Initializing components...")
    collector = NBADataCollector()
    engineer = FeatureEngineer()
    predictor = NBAPredictor()
    
    print("Collecting historical game data...")
    historical_games = collector.get_historical_games()
    
    print("Preparing training data...")
    X, y = predictor.prepare_training_data(historical_games)
    
    print("Training model...")
    predictor.train(X, y)
    
    print("Model training complete!")
    print("The model has been saved to the 'models' directory.")
    
if __name__ == "__main__":
    main() 
