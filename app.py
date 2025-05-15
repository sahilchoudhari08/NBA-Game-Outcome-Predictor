import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

from data_collection import NBADataCollector
from feature_engineering import FeatureEngineer
from model import NBAPredictor
from visualization import NBAVisualizer

# Set page config
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return {
        'collector': NBADataCollector(),
        'engineer': FeatureEngineer(),
        'predictor': NBAPredictor(),
        'visualizer': NBAVisualizer()
    }

components = init_components()

# App title and description
st.title("🏀 NBA Game Outcome Predictor")
st.markdown("""
This app predicts the outcome of NBA games using historical data and current team statistics.
Select two teams to see the prediction and detailed analysis.
""")

# Sidebar for team selection
st.sidebar.header("Team Selection")

# Get list of teams
teams = components['collector'].teams
team_names = [team['full_name'] for team in teams]

# Team selection dropdowns
home_team = st.sidebar.selectbox("Home Team", team_names)
away_team = st.sidebar.selectbox("Away Team", [t for t in team_names if t != home_team])

# Get team IDs
home_team_id = components['collector'].team_ids[home_team]
away_team_id = components['collector'].team_ids[away_team]

# Main content
if st.sidebar.button("Predict Game Outcome"):
    with st.spinner("Analyzing teams and making prediction..."):
        # Get team data
        home_team_games = components['collector'].get_team_stats(home_team_id)
        away_team_games = components['collector'].get_team_stats(away_team_id)
        print(f"Home team ({home_team}) game logs:")
        print(home_team_games)
        print(f"Away team ({away_team}) game logs:")
        print(away_team_games)
        if home_team_games.empty or away_team_games.empty:
            st.warning("One or both teams have no recent game data. Predictions may not be meaningful.")
        historical_games = components['collector'].get_historical_games()
        
        # Add OPPONENT_TEAM_ID column if missing
        if 'OPPONENT_TEAM_ID' not in historical_games.columns:
            def get_opponent_team_id(row):
                # Parse the MATCHUP string to get the opponent abbreviation
                if 'vs.' in row['MATCHUP']:
                    # Home game: 'ATL vs. BOS'
                    opp_abbr = row['MATCHUP'].split('vs.')[1].strip()
                elif '@' in row['MATCHUP']:
                    # Away game: 'ATL @ BOS'
                    opp_abbr = row['MATCHUP'].split('@')[1].strip()
                else:
                    return None
                # Map abbreviation to team ID
                for team in teams:
                    if team['abbreviation'] == opp_abbr:
                        return team['id']
                return None
            historical_games['OPPONENT_TEAM_ID'] = historical_games.apply(get_opponent_team_id, axis=1)
        
        # Prepare features
        features = components['engineer'].prepare_features(
            home_team_games,
            away_team_games
        )
        print("Features for prediction:")
        print(features)
        
        # Make prediction
        prediction = components['predictor'].predict(features)
        
        # Display prediction
        st.header("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predicted Winner")
            winner = home_team if prediction['prediction'] == 1 else away_team
            st.success(f"🏆 {winner}")
            if prediction.get('warning'):
                st.warning(prediction['warning'])
        
        with col2:
            st.subheader("Confidence Score")
            probability = prediction['probability']
            st.metric(
                "Win Probability",
                f"{probability:.1%}",
                f"{'Home' if prediction['prediction'] == 1 else 'Away'} Team"
            )
            st.caption("This is the model's estimated probability that the predicted team will win, based on recent team performance, scoring, and advanced stats.")
        
        # Recent form
        st.subheader("Recent Form")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home_team} Last 5 Games:**")
            cols = [c for c in ['GAME_DATE', 'WL', 'PTS', 'PLUS_MINUS'] if c in home_team_games.columns]
            if cols:
                st.dataframe(home_team_games.head(5)[cols])
            else:
                st.info("No recent game data available.")
        with col2:
            st.write(f"**{away_team} Last 5 Games:**")
            cols = [c for c in ['GAME_DATE', 'WL', 'PTS', 'PLUS_MINUS'] if c in away_team_games.columns]
            if cols:
                st.dataframe(away_team_games.head(5)[cols])
            else:
                st.info("No recent game data available.")
        # Head-to-head history as a table
        st.header("Head-to-Head History")
        h2h_games = historical_games[
            ((historical_games['TEAM_ID'] == home_team_id) & 
             (historical_games['OPPONENT_TEAM_ID'] == away_team_id)) |
            ((historical_games['TEAM_ID'] == away_team_id) & 
             (historical_games['OPPONENT_TEAM_ID'] == home_team_id))
        ]
        if not h2h_games.empty:
            h2h_games = h2h_games.sort_values('GAME_DATE', ascending=False).head(5)
            # Build a summary table
            def get_opp_pts(row):
                opp_row = h2h_games[(h2h_games['GAME_ID'] == row['GAME_ID']) & (h2h_games['TEAM_ID'] != row['TEAM_ID'])]
                if not opp_row.empty:
                    return opp_row.iloc[0]['PTS']
                return 'N/A'
            def get_score(row):
                opp_pts = get_opp_pts(row)
                return f"{row['PTS']} - {opp_pts}" if row['TEAM_ID'] == home_team_id else f"{opp_pts} - {row['PTS']}"
            h2h_table = pd.DataFrame({
                'Date': h2h_games['GAME_DATE'].dt.strftime('%Y-%m-%d'),
                'Home Team': [home_team if row['MATCHUP'].split(' ')[1] == 'vs.' else away_team for _, row in h2h_games.iterrows()],
                'Away Team': [away_team if row['MATCHUP'].split(' ')[1] == 'vs.' else home_team for _, row in h2h_games.iterrows()],
                'Score': [get_score(row) for _, row in h2h_games.iterrows()],
                'Result': h2h_games['WL']
            })
            st.dataframe(h2h_table)
        else:
            st.info("No head-to-head history available for these teams.")
            
        # Additional information
        st.header("Additional Information")
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        **Disclaimer:** This prediction is based on historical data and statistical analysis.
        Many factors can influence the outcome of a game, and this prediction should not be
        used as the sole basis for any betting or decision-making.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and NBA Stats API</p>
    <p>Data last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True) 
