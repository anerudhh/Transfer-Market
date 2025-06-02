import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class FootballDataLoader:
    def __init__(self):
        self.players_df = None
        self.appearances_df = None
        self.valuations_df = None
        self.clubs_df = None
        self.competitions_df = None
        self.load_data()
    
    def load_data(self):
        """Load all CSV files into dataframes"""
        try:
            print("Loading football data...")
            self.players_df = pd.read_csv('attached_assets/players.csv')
            self.appearances_df = pd.read_csv('attached_assets/appearances.csv')
            self.valuations_df = pd.read_csv('attached_assets/player_valuations.csv')
            self.clubs_df = pd.read_csv('attached_assets/clubs.csv')
            self.competitions_df = pd.read_csv('attached_assets/competitions.csv')
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_top_5_leagues(self):
        """Get competition IDs for top 5 European leagues"""
        top_leagues = {
            'GB1': 'Premier League',
            'ES1': 'La Liga', 
            'IT1': 'Serie A',
            'L1': 'Bundesliga',
            'FR1': 'Ligue 1'
        }
        return top_leagues
    
    def search_player(self, name: str) -> Optional[Dict[str, Any]]:
        """Search for a player by name in the real dataset"""
        if self.players_df is None:
            return None
        
        # Search for player (case insensitive)
        mask = self.players_df['name'].str.contains(name, case=False, na=False)
        matches = self.players_df[mask]
        
        if matches.empty:
            return None
        
        # Get the first match
        player = matches.iloc[0]
        player_id = player['player_id']
        
        # Get player's latest market value
        player_valuations = self.valuations_df[self.valuations_df['player_id'] == player_id]
        if not player_valuations.empty:
            latest_valuation = player_valuations.loc[player_valuations['datetime'].idxmax()]
            market_value = latest_valuation['market_value_in_eur']
        else:
            market_value = player.get('market_value_in_eur', 0)
        
        # Get player's career statistics
        player_appearances = self.appearances_df[self.appearances_df['player_id'] == player_id]
        
        # Calculate career totals
        total_appearances = len(player_appearances)
        total_goals = player_appearances['goals'].sum() if not player_appearances.empty else 0
        total_assists = player_appearances['assists'].sum() if not player_appearances.empty else 0
        total_minutes = player_appearances['minutes_played'].sum() if not player_appearances.empty else 0
        
        # Get club name
        club_name = player.get('current_club_name', 'Unknown')
        
        # Get league name
        competition_id = player.get('current_club_domestic_competition_id', '')
        top_leagues = self.get_top_5_leagues()
        league_name = top_leagues.get(competition_id, 'Other League')
        
        # Calculate age from date of birth
        try:
            birth_date = pd.to_datetime(player['date_of_birth'])
            age = (pd.Timestamp.now() - birth_date).days // 365
        except:
            age = None
        
        return {
            'player_id': player_id,
            'name': player['name'],
            'age': age,
            'position': player.get('position', 'Unknown'),
            'current_club': club_name,
            'league': league_name,
            'appearances': int(total_appearances),
            'goals': int(total_goals),
            'assists': int(total_assists),
            'minutes_played': int(total_minutes),
            'market_value': market_value if pd.notna(market_value) else 0,
            'height': player.get('height_in_cm', None),
            'foot': player.get('foot', 'Unknown'),
            'country': player.get('country_of_citizenship', 'Unknown')
        }
    
    def get_top_5_leagues_players(self, limit=1000):
        """Get players from top 5 European leagues for training"""
        top_leagues = self.get_top_5_leagues()
        
        # Filter players from top 5 leagues
        top_league_players = self.players_df[
            self.players_df['current_club_domestic_competition_id'].isin(top_leagues.keys())
        ].copy()
        
        if len(top_league_players) > limit:
            top_league_players = top_league_players.sample(n=limit, random_state=42)
        
        training_data = []
        
        for _, player in top_league_players.iterrows():
            player_id = player['player_id']
            
            # Get latest market value
            player_valuations = self.valuations_df[self.valuations_df['player_id'] == player_id]
            if not player_valuations.empty:
                latest_valuation = player_valuations.loc[player_valuations['datetime'].idxmax()]
                market_value = latest_valuation['market_value_in_eur']
            else:
                market_value = player.get('market_value_in_eur', 0)
            
            # Skip players without market value
            if pd.isna(market_value) or market_value <= 0:
                continue
            
            # Get career statistics
            player_appearances = self.appearances_df[self.appearances_df['player_id'] == player_id]
            
            total_appearances = len(player_appearances)
            total_goals = player_appearances['goals'].sum() if not player_appearances.empty else 0
            total_assists = player_appearances['assists'].sum() if not player_appearances.empty else 0
            total_minutes = player_appearances['minutes_played'].sum() if not player_appearances.empty else 0
            
            # Calculate age
            try:
                birth_date = pd.to_datetime(player['date_of_birth'])
                age = (pd.Timestamp.now() - birth_date).days // 365
            except:
                continue  # Skip players without valid birth date
            
            # Skip players with invalid age or no appearances
            if age < 16 or age > 45 or total_appearances == 0:
                continue
            
            training_data.append({
                'age': age,
                'position': player.get('position', 'Unknown'),
                'league': top_leagues[player['current_club_domestic_competition_id']],
                'appearances': total_appearances,
                'goals': total_goals,
                'assists': total_assists,
                'minutes_played': total_minutes,
                'market_value': market_value
            })
        
        return pd.DataFrame(training_data)
    
    def get_all_player_names(self, limit=100):
        """Get a list of player names for suggestions"""
        if self.players_df is None:
            return []
        
        # Get players from top 5 leagues with market values
        top_leagues = self.get_top_5_leagues()
        top_players = self.players_df[
            (self.players_df['current_club_domestic_competition_id'].isin(top_leagues.keys())) &
            (self.players_df['market_value_in_eur'].notna()) &
            (self.players_df['market_value_in_eur'] > 0)
        ].copy()
        
        # Sort by market value and get top players
        top_players = top_players.sort_values('market_value_in_eur', ascending=False)
        
        return top_players['name'].head(limit).tolist()