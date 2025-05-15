import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, teamgamelog
from nba_api.stats.static import teams
import joblib
from pathlib import Path
import time

class NBADataCollector:
    def __init__(self, cache_dir='data'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.teams = teams.get_teams()
        self.team_ids = {team['full_name']: team['id'] for team in self.teams}
        
    def _get_cache_path(self, filename):
        return self.cache_dir / filename
        
    def _load_cached_data(self, filename):
        cache_path = self._get_cache_path(filename)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return None
        
    def _save_to_cache(self, data, filename):
        cache_path = self._get_cache_path(filename)
        data.to_csv(cache_path, index=False)
        
    def get_team_game_logs(self, team_id, season='2023-24', last_n_games=20):
        """Get game logs for a specific team"""
        try:
            # Get game logs
            game_logs = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                last_n_games=last_n_games
            ).get_data_frames()[0]
            
            # Ensure all required columns are present
            required_columns = [
                'GAME_ID', 'GAME_DATE', 'WL', 'PTS', 'OPP_PTS',
                'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING'
            ]
            
            # Add missing columns with default values if necessary
            for col in required_columns:
                if col not in game_logs.columns:
                    if col in ['WL', 'GAME_ID', 'GAME_DATE']:
                        game_logs[col] = None
                    else:
                        game_logs[col] = 0.0
            
            # Sort by date
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
            game_logs = game_logs.sort_values('GAME_DATE', ascending=False)
            
            return game_logs
            
        except Exception as e:
            print(f"Error getting game logs for team {team_id}: {str(e)}")
            return pd.DataFrame()
        
    def get_historical_games(self, season='2023-24'):
        """Get all games for the current season"""
        try:
            # Get all games
            games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season
            ).get_data_frames()[0]
            
            # Sort by date
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games = games.sort_values('GAME_DATE', ascending=False)
            
            return games
            
        except Exception as e:
            print(f"Error getting historical games: {str(e)}")
            return pd.DataFrame()
        
    def get_team_id(self, team_name):
        """Get team ID from team name"""
        return self.team_ids.get(team_name)
        
    def get_all_teams(self):
        """Get list of all NBA teams"""
        return list(self.team_ids.keys())
        
    def get_team_stats(self, team_id, last_n_games=10):
        """Get recent team statistics"""
        cache_file = f'team_stats_{team_id}_{last_n_games}.csv'
        cached_data = self._load_cached_data(cache_file)
        
        if cached_data is not None:
            return cached_data
            
        team_games = teamgamelog.TeamGameLog(
            team_id=team_id,
            season='2023-24'
        ).get_data_frames()[0]
        
        team_games = team_games.head(last_n_games)
        self._save_to_cache(team_games, cache_file)
        return team_games
        
    def get_player_stats(self, player_id, last_n_games=10):
        """Get recent player statistics"""
        cache_file = f'player_stats_{player_id}_{last_n_games}.csv'
        cached_data = self._load_cached_data(cache_file)
        
        if cached_data is not None:
            return cached_data
            
        player_games = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2023-24'
        ).get_data_frames()[0]
        
        player_games = player_games.head(last_n_games)
        self._save_to_cache(player_games, cache_file)
        return player_games
        
    def get_team_roster(self, team_id):
        """Get current team roster"""
        cache_file = f'roster_{team_id}.csv'
        cached_data = self._load_cached_data(cache_file)
        
        if cached_data is not None:
            return cached_data
            
        # Implementation for getting team roster
        # This would need to be implemented using the appropriate NBA API endpoint
        pass
        
    def update_all_data(self):
        """Update all cached data"""
        self.get_historical_games()
        for team_id in self.team_ids.values():
            self.get_team_stats(team_id)
            self.get_team_roster(team_id)
            
if __name__ == "__main__":
    collector = NBADataCollector()
    collector.update_all_data() 
