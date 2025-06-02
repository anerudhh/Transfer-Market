import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from data_loader import FootballDataLoader

# Configure page
st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

@st.cache_resource
def load_data_loader():
    """Load the football data loader with caching"""
    return FootballDataLoader()

@st.cache_resource  
def load_model():
    """Load the pre-trained model with caching for better performance."""
    try:
        with open('real_market_value_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        # If real model doesn't exist, use the synthetic one
        try:
            with open('market_value_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return {'model': model, 'position_encoder': None, 'league_encoder': None}
        except FileNotFoundError:
            st.error("No model file found. Please train a model first.")
            st.stop()

def predict_market_value_real(player_data: Dict[str, Any], model_data) -> float:
    """Predict market value using real player data and trained model."""
    model = model_data['model']
    
    # If we have encoders (real model), use them
    if model_data.get('position_encoder') and model_data.get('league_encoder'):
        position_encoder = model_data['position_encoder']
        league_encoder = model_data['league_encoder']
        
        # Encode categorical variables
        try:
            position_encoded = position_encoder.transform([player_data['position']])[0]
        except ValueError:
            # If position not seen in training, use most common encoding
            position_encoded = 0
            
        try:
            league_encoded = league_encoder.transform([player_data['league']])[0]
        except ValueError:
            # If league not seen in training, use most common encoding
            league_encoded = 0
    else:
        # Fallback to manual encoding for synthetic model
        position_encoding = {'Attack': 3, 'Midfield': 2, 'Defender': 1, 'Goalkeeper': 0}
        league_encoding = {'Premier League': 4, 'La Liga': 3, 'Serie A': 2, 'Bundesliga': 1, 'Ligue 1': 0}
        
        position_encoded = position_encoding.get(player_data['position'], 1)
        league_encoded = league_encoding.get(player_data['league'], 1)
    
    # Prepare features
    features = np.array([[
        player_data['age'],
        position_encoded,
        league_encoded,
        player_data['appearances'],
        player_data['goals'],
        player_data['assists'],
        player_data['minutes_played']
    ]])
    
    prediction = model.predict(features)[0]
    return max(prediction, 0)

def display_feature_importance(model_data):
    """Display feature importance chart."""
    model = model_data['model']
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
    st.markdown("Search for real football players and predict their market value using machine learning!")
    
    # Load data and model
    data_loader = load_data_loader()
    model_data = load_model()
    
    # Get popular player names for suggestions
    try:
        popular_players = data_loader.get_all_player_names(50)
    except:
        popular_players = []
    
    # User input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        player_name = st.text_input(
            "Search for a Football Player:",
            placeholder="e.g., Lionel Messi, Cristiano Ronaldo, Kylian Mbappé",
            help="Enter the name of any football player from the top European leagues"
        )
    
    with col2:
        predict_button = st.button("🔮 Predict Market Value", type="primary")
    
    # Search suggestions
    if player_name and len(player_name) > 2:
        matching_players = [p for p in popular_players if player_name.lower() in p.lower()]
        if matching_players:
            st.write("**Suggestions:**")
            suggestion_cols = st.columns(min(5, len(matching_players)))
            for i, suggestion in enumerate(matching_players[:5]):
                with suggestion_cols[i]:
                    if st.button(suggestion, key=f"suggest_{i}"):
                        st.session_state.selected_player = suggestion
                        st.rerun()
    
    # Use selected player from suggestions
    if 'selected_player' in st.session_state:
        player_name = st.session_state.selected_player
        del st.session_state.selected_player
        predict_button = True
    
    if player_name and predict_button:
        # Search for player in real data
        with st.spinner("Searching for player in database..."):
            player_data = data_loader.search_player(player_name)
        
        if player_data is None:
            st.error(f"Player '{player_name}' not found in the database. Please try a different name or check the spelling.")
            st.info("Try searching for players from Premier League, La Liga, Serie A, Bundesliga, or Ligue 1.")
            return
        
        # Display player information
        st.success(f"Found player: **{player_data['name']}**")
        st.subheader(f"📊 Player Profile: {player_data['name']}")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Player basic info
            st.markdown("### Basic Information")
            basic_info_data = []
            if player_data['age']:
                basic_info_data.append(["Age", f"{player_data['age']} years"])
            basic_info_data.extend([
                ["Position", player_data['position']],
                ["Current Club", player_data['current_club']],
                ["League", player_data['league']],
                ["Country", player_data['country']]
            ])
            
            if player_data['height']:
                basic_info_data.append(["Height", f"{player_data['height']} cm"])
            if player_data['foot'] != 'Unknown':
                basic_info_data.append(["Preferred Foot", player_data['foot']])
            
            basic_df = pd.DataFrame(basic_info_data, columns=["Attribute", "Value"])
            st.table(basic_df)
        
        with col2:
            # Career statistics
            st.markdown("### Career Statistics")
            stats_data = [
                ["Total Appearances", f"{player_data['appearances']:,}"],
                ["Total Goals", f"{player_data['goals']:,}"],
                ["Total Assists", f"{player_data['assists']:,}"],
                ["Total Minutes", f"{player_data['minutes_played']:,}"]
            ]
            
            # Calculate additional metrics if possible
            if player_data['appearances'] > 0:
                goals_per_game = player_data['goals'] / player_data['appearances']
                assists_per_game = player_data['assists'] / player_data['appearances']
                stats_data.extend([
                    ["Goals per Game", f"{goals_per_game:.2f}"],
                    ["Assists per Game", f"{assists_per_game:.2f}"]
                ])
            
            stats_df = pd.DataFrame(stats_data, columns=["Statistic", "Value"])
            st.table(stats_df)
        
        # Predict market value
        st.subheader("💰 Market Value Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Show actual market value if available
            if player_data['market_value'] > 0:
                st.metric(
                    "Actual Market Value (Database)",
                    f"€{player_data['market_value']:,.0f}",
                    help="Market value from the database"
                )
        
        with col2:
            # Predict market value using ML model
            if player_data['age'] and player_data['appearances'] > 0:
                predicted_value = predict_market_value_real(player_data, model_data)
                
                st.metric(
                    "Predicted Market Value (ML Model)",
                    f"€{predicted_value:,.0f}",
                    help="Market value predicted by machine learning model"
                )
                
                # Calculate difference if we have actual value
                if player_data['market_value'] > 0:
                    difference = predicted_value - player_data['market_value']
                    percentage_diff = (difference / player_data['market_value']) * 100
                    
                    if abs(percentage_diff) < 20:
                        st.success(f"Model prediction is within 20% of actual value ({percentage_diff:+.1f}%)")
                    elif abs(percentage_diff) < 50:
                        st.info(f"Model prediction differs by {abs(percentage_diff):.1f}% from actual value")
                    else:
                        st.warning(f"Model prediction differs significantly from actual value ({percentage_diff:+.1f}%)")
            else:
                st.warning("Cannot predict market value: insufficient player data")
        
        # Add value context
        if 'predicted_value' in locals():
            if predicted_value > 100_000_000:
                st.success("🌟 World-class player!")
            elif predicted_value > 50_000_000:
                st.info("⭐ Elite player!")
            elif predicted_value > 20_000_000:
                st.info("🔥 High-value player!")
            else:
                st.info("📈 Developing or veteran player!")
        
        # Feature importance visualization
        st.subheader("📈 Model Feature Importance")
        st.markdown("This chart shows which statistics most influence the market value prediction:")
        
        fig = display_feature_importance(model_data)
        st.pyplot(fig)
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Model Type:** Random Forest Regressor")
            st.write(f"**Number of Features:** {len(model_data['model'].feature_importances_)}")
            st.write(f"**Training Data:** Real player data from European football leagues")
            st.write("**Features Used:** Age, Position, League, Appearances, Goals, Assists, Minutes Played")
    
    elif player_name and not predict_button:
        st.info("👆 Click the 'Predict Market Value' button to search for the player!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses real football player data to predict market values using machine learning.
        
        **How it works:**
        1. Enter any player name from top European leagues
        2. The app searches the real database
        3. ML model predicts market value based on career statistics
        4. Compare with actual market values
        
        **Data Sources:**
        - Real player statistics and market values
        - Career appearance and performance data
        - Players from Premier League, La Liga, Serie A, Bundesliga, and Ligue 1
        """)
        
        st.header("🏆 Popular Players to Try:")
        if popular_players:
            for i, player in enumerate(popular_players[:8]):
                if st.button(player, key=f"sidebar_{i}"):
                    st.session_state.selected_player = player
                    st.rerun()
        
        st.header("🔍 Search Tips")
        st.write("""
        - Use full player names for best results
        - Try variations if not found initially
        - Search works for current and recent players
        - Focus on players from top 5 European leagues
        """)

if __name__ == "__main__":
    main()