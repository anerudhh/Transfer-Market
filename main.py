import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

def get_player_stats(name: str) -> Dict[str, Any]:
    """
    Placeholder function that returns synthetic player statistics.
    In a real application, this would fetch data from a database or API.
    """
    # Define realistic ranges for different positions and leagues
    positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    leagues = ['Premier League', 'La Liga', 'Serie A', 'Ligue 1', 'Bundesliga']
    
    # Clubs for each league
    clubs = {
        'Premier League': ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United', 'Tottenham'],
        'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Real Sociedad'],
        'Serie A': ['Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio'],
        'Ligue 1': ['PSG', 'Monaco', 'Marseille', 'Lyon', 'Lille', 'Nice'],
        'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt', 'Wolfsburg']
    }
    
    # Generate consistent stats based on name hash for reproducibility
    random.seed(hash(name) % (2**32))
    
    position = random.choice(positions)
    league = random.choice(leagues)
    club = random.choice(clubs[league])
    age = random.randint(18, 38)
    
    # Adjust stats based on position
    if position == 'Goalkeeper':
        appearances = random.randint(15, 38)
        goals = random.randint(0, 2)
        assists = random.randint(0, 3)
        minutes_played = appearances * random.randint(80, 90)
    elif position == 'Forward':
        appearances = random.randint(20, 38)
        goals = random.randint(5, 35)
        assists = random.randint(2, 15)
        minutes_played = appearances * random.randint(70, 90)
    elif position == 'Midfielder':
        appearances = random.randint(25, 38)
        goals = random.randint(2, 20)
        assists = random.randint(5, 20)
        minutes_played = appearances * random.randint(75, 90)
    else:  # Defender
        appearances = random.randint(25, 38)
        goals = random.randint(0, 8)
        assists = random.randint(1, 10)
        minutes_played = appearances * random.randint(80, 90)
    
    return {
        'age': age,
        'position': position,
        'current_club': club,
        'league': league,
        'appearances': appearances,
        'goals': goals,
        'assists': assists,
        'minutes_played': minutes_played
    }

@st.cache_resource
def load_model():
    """Load the pre-trained model with caching for better performance."""
    try:
        with open('market_value_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'market_value_model.pkl' not found. Please run 'train_model.py' first to create the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_market_value(stats: Dict[str, Any], model) -> float:
    """Predict market value based on player statistics."""
    # Convert categorical variables to numerical
    position_encoding = {'Forward': 3, 'Midfielder': 2, 'Defender': 1, 'Goalkeeper': 0}
    league_encoding = {'Premier League': 4, 'La Liga': 3, 'Serie A': 2, 'Bundesliga': 1, 'Ligue 1': 0}
    
    # Prepare features in the same order as training
    features = np.array([[
        stats['age'],
        position_encoding[stats['position']],
        league_encoding[stats['league']],
        stats['appearances'],
        stats['goals'],
        stats['assists'],
        stats['minutes_played']
    ]])
    
    prediction = model.predict(features)[0]
    return max(prediction, 0)  # Ensure non-negative values

def display_feature_importance(model):
    """Display feature importance chart."""
    feature_names = ['Age', 'Position', 'League', 'Appearances', 'Goals', 'Assists', 'Minutes Played']
    importance = model.feature_importances_
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, importance, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance in Market Value Prediction')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    # App header
    st.title("⚽ Football Player Market Value Predictor")
    st.markdown("Enter a player's name to predict their market value using machine learning!")
    
    # Load model
    model = load_model()
    
    # User input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        player_name = st.text_input(
            "Enter Player Name:",
            placeholder="e.g., Lionel Messi, Cristiano Ronaldo, Kylian Mbappé"
        )
    
    with col2:
        predict_button = st.button("🔮 Predict Market Value", type="primary")
    
    if player_name and predict_button:
        # Get player stats
        with st.spinner("Analyzing player statistics..."):
            stats = get_player_stats(player_name)
        
        # Display player information
        st.subheader(f"📊 Player Profile: {player_name}")
        
        # Create stats dataframe for display
        stats_df = pd.DataFrame([
            ["Age", stats['age']],
            ["Position", stats['position']],
            ["Current Club", stats['current_club']],
            ["League", stats['league']],
            ["Appearances", stats['appearances']],
            ["Goals", stats['goals']],
            ["Assists", stats['assists']],
            ["Minutes Played", f"{stats['minutes_played']:,}"]
        ], columns=["Statistic", "Value"])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.table(stats_df)
        
        with col2:
            # Predict market value
            predicted_value = predict_market_value(stats, model)
            
            st.metric(
                "💰 Predicted Market Value",
                f"€{predicted_value:,.0f}",
                help="Predicted market value in Euros"
            )
            
            # Add some context based on value ranges
            if predicted_value > 100_000_000:
                st.success("🌟 World-class player!")
            elif predicted_value > 50_000_000:
                st.info("⭐ Elite player!")
            elif predicted_value > 20_000_000:
                st.info("🔥 High-value player!")
            else:
                st.info("📈 Developing player!")
        
        # Feature importance visualization
        st.subheader("📈 Model Feature Importance")
        st.markdown("This chart shows which statistics most influence the market value prediction:")
        
        fig = display_feature_importance(model)
        st.pyplot(fig)
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Model Type:** Random Forest Regressor")
            st.write(f"**Number of Features:** {len(model.feature_importances_)}")
            st.write(f"**Training Data:** Synthetic dataset from Top 5 European leagues")
            st.write("**Features Used:** Age, Position, League, Appearances, Goals, Assists, Minutes Played")
    
    elif player_name and not predict_button:
        st.info("👆 Click the 'Predict Market Value' button to see the prediction!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app predicts football player market values using a Random Forest machine learning model.
        
        **How it works:**
        1. Enter any player name
        2. The app generates realistic statistics
        3. ML model predicts market value
        4. View feature importance analysis
        
        **Leagues covered:**
        - Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿
        - La Liga 🇪🇸
        - Serie A 🇮🇹
        - Bundesliga 🇩🇪
        - Ligue 1 🇫🇷
        """)
        
        st.header("🎯 Try these players:")
        example_players = [
            "Lionel Messi",
            "Kylian Mbappé",
            "Erling Haaland",
            "Pedri",
            "Jude Bellingham"
        ]
        
        for player in example_players:
            if st.button(player, key=f"example_{player}"):
                st.session_state.player_name = player
                st.rerun()

if __name__ == "__main__":
    main()
