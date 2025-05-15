import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class NBAVisualizer:
    def __init__(self):
        self.colors = {
            'home': '#1f77b4',
            'away': '#ff7f0e',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        }
        
    def create_team_comparison_chart(self, home_stats, away_stats, stat_names):
        """Create a bar chart comparing team statistics"""
        fig = go.Figure()
        
        # Add home team bars
        fig.add_trace(go.Bar(
            name='Home Team',
            x=stat_names,
            y=[home_stats.get(stat, 0) for stat in stat_names],
            marker_color=self.colors['home']
        ))
        
        # Add away team bars
        fig.add_trace(go.Bar(
            name='Away Team',
            x=stat_names,
            y=[away_stats.get(stat, 0) for stat in stat_names],
            marker_color=self.colors['away']
        ))
        
        # Update layout
        fig.update_layout(
            title='Team Statistics Comparison',
            barmode='group',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=True
        )
        
        return fig
        
    def create_win_probability_gauge(self, probability, team_name):
        """Create a gauge chart showing win probability"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': f"{team_name} Win Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['home']},
                'steps': [
                    {'range': [0, 50], 'color': self.colors['away']},
                    {'range': [50, 100], 'color': self.colors['home']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
        
    def create_feature_importance_chart(self, feature_importance):
        """Create a horizontal bar chart showing feature importance"""
        # Sort features by importance
        sorted_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        # Only keep top 5 and filter out zero/None importances
        top_features = [(k, v) for k, v in sorted_features.items() if v and v > 0][:5]
        if not top_features:
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_annotation(text="No significant features found for this prediction.",
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=16))
            fig.update_layout(title='Feature Importance')
            return fig
        # Human-friendly feature names
        def pretty_name(name):
            return (name.replace('_', ' ').replace('pct', '%')
                        .replace('home ', 'Home ').replace('away ', 'Away ')
                        .replace('win', 'Win').replace('losses', 'Losses')
                        .replace('points', 'Points').replace('per game', '/Game')
                        .replace('recent', 'Recent ')
                        .replace('fg', 'FG').replace('ft', 'FT').replace('3', '3'))
        labels = [pretty_name(k) for k, v in top_features]
        values = [v for k, v in top_features]
        import plotly.graph_objs as go
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=self.colors['home']
        ))
        fig.update_layout(
            title='Top 5 Most Influential Features',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        return fig
        
    def create_h2h_history_chart(self, h2h_games):
        """Create a line chart showing head-to-head history"""
        # Convert game dates to datetime
        h2h_games['GAME_DATE'] = pd.to_datetime(h2h_games['GAME_DATE'])
        h2h_games = h2h_games.sort_values('GAME_DATE')
        
        # Calculate cumulative wins
        h2h_games['CUMULATIVE_WINS'] = h2h_games['WL'].eq('W').cumsum()
        
        fig = go.Figure()
        
        # Add line for home team
        fig.add_trace(go.Scatter(
            x=h2h_games['GAME_DATE'],
            y=h2h_games['CUMULATIVE_WINS'],
            mode='lines+markers',
            name='Home Team',
            line=dict(color=self.colors['home'])
        ))
        
        fig.update_layout(
            title='Head-to-Head History',
            xaxis_title='Date',
            yaxis_title='Cumulative Wins',
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig
        
    def create_player_comparison_radar(self, home_players, away_players):
        """Create a radar chart comparing top players"""
        categories = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
        
        fig = go.Figure()
        
        # Add home team players
        for player in home_players:
            fig.add_trace(go.Scatterpolar(
                r=[player['PTS'], player['REB'], player['AST'],
                   player['STL'], player['BLK']],
                theta=categories,
                fill='toself',
                name=f"Home: {player['PLAYER_NAME']}"
            ))
            
        # Add away team players
        for player in away_players:
            fig.add_trace(go.Scatterpolar(
                r=[player['PTS'], player['REB'], player['AST'],
                   player['STL'], player['BLK']],
                theta=categories,
                fill='toself',
                name=f"Away: {player['PLAYER_NAME']}"
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(
                        max(p['PTS'] for p in home_players + away_players),
                        max(p['REB'] for p in home_players + away_players),
                        max(p['AST'] for p in home_players + away_players),
                        max(p['STL'] for p in home_players + away_players),
                        max(p['BLK'] for p in home_players + away_players)
                    )]
                )
            ),
            showlegend=True,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'])
        )
        
        return fig 