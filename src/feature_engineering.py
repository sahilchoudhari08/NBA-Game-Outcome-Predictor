import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = None
        
    def calculate_team_stats(self, team_games, is_home=None):
        """Calculate team statistics from game data, with home/away splits if possible"""
        stats = {}
        
        # Basic stats
        stats['wins'] = (team_games['WL'] == 'W').sum()
        stats['losses'] = (team_games['WL'] == 'L').sum()
        stats['win_pct'] = stats['wins'] / len(team_games) if len(team_games) > 0 else 0
        
        # Season-long stats (use all games)
        stats['season_win_pct'] = stats['wins'] / len(team_games) if len(team_games) > 0 else 0
        stats['season_net_rating'] = team_games['NET_RATING'].mean() if 'NET_RATING' in team_games.columns else 0
        
        # Home/away splits
        if is_home is not None and 'MATCHUP' in team_games.columns:
            home_games = team_games[team_games['MATCHUP'].str.contains('vs.')]
            away_games = team_games[team_games['MATCHUP'].str.contains('@')]
            stats['win_pct_at_home'] = (home_games['WL'] == 'W').mean() if len(home_games) > 0 else 0
            stats['win_pct_on_road'] = (away_games['WL'] == 'W').mean() if len(away_games) > 0 else 0
            stats['net_rating_at_home'] = home_games['NET_RATING'].mean() if 'NET_RATING' in home_games.columns and len(home_games) > 0 else 0
            stats['net_rating_on_road'] = away_games['NET_RATING'].mean() if 'NET_RATING' in away_games.columns and len(away_games) > 0 else 0
            stats['is_home_team'] = int(is_home)
        
        # Recent form (last 20 games)
        recent_games = team_games.head(20)
        stats['recent_win_pct'] = (recent_games['WL'] == 'W').mean() if 'WL' in recent_games.columns and len(recent_games) > 0 else 0
        stats['recent_net_rating'] = recent_games['NET_RATING'].mean() if 'NET_RATING' in recent_games.columns and len(recent_games) > 0 else 0
        
        # Scoring stats
        stats['points_per_game'] = team_games['PTS'].mean() if 'PTS' in team_games.columns else 0
        if 'OPP_PTS' in team_games.columns:
            stats['points_allowed'] = team_games['OPP_PTS'].mean()
        elif 'PTS' in team_games.columns and 'PLUS_MINUS' in team_games.columns:
            stats['points_allowed'] = (team_games['PTS'] - team_games['PLUS_MINUS']).mean()
        else:
            stats['points_allowed'] = 0
        stats['point_diff'] = (stats['points_per_game'] or 0) - (stats['points_allowed'] or 0)
        
        # Shooting stats
        stats['fg_pct'] = team_games['FG_PCT'].mean() if 'FG_PCT' in team_games.columns else 0
        stats['fg3_pct'] = team_games['FG3_PCT'].mean() if 'FG3_PCT' in team_games.columns else 0
        stats['ft_pct'] = team_games['FT_PCT'].mean() if 'FT_PCT' in team_games.columns else 0
        
        # Advanced stats
        stats['off_rating'] = team_games['OFF_RATING'].mean() if 'OFF_RATING' in team_games.columns else 0
        stats['def_rating'] = team_games['DEF_RATING'].mean() if 'DEF_RATING' in team_games.columns else 0
        stats['net_rating'] = team_games['NET_RATING'].mean() if 'NET_RATING' in team_games.columns else 0
        
        return stats
        
    def prepare_features(self, home_team_games, away_team_games):
        """Prepare features for model input, with home/away splits and is_home_team"""
        # Calculate team stats
        home_stats = self.calculate_team_stats(home_team_games, is_home=True)
        away_stats = self.calculate_team_stats(away_team_games, is_home=False)
        
        # Create feature dictionary
        features = {}
        
        # Home team features
        for stat, value in home_stats.items():
            features[f'home_{stat}'] = value
            
        # Away team features
        for stat, value in away_stats.items():
            features[f'away_{stat}'] = value
            
        # Add a strong binary home_court_advantage feature
        features['home_court_advantage'] = 1
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        return features_df
        
    def get_feature_columns(self):
        """Get list of feature columns"""
        if self.feature_columns is None:
            raise ValueError("Features haven't been prepared yet")
        return self.feature_columns 
