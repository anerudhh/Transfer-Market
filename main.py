import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Configure page
st.set_page_config(
    page_title="Football Player Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

@st.cache_resource
def load_players():
    df = pd.read_csv('attached_assets/players.csv')
    df.columns = df.columns.str.strip().str.lower()
    df['name_lower'] = df['name'].str.lower()
    return df

@st.cache_resource
def load_valuations():
    df = pd.read_csv('attached_assets/player_valuations.csv')
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_resource
def load_game_events():
    df = pd.read_csv('game_events.csv')
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_resource
def load_appearances():
    try:
        df = pd.read_csv('attached_assets/appearances.csv')
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return None

def get_latest_market_value(player_id, valuations_df):
    vals = valuations_df[valuations_df['player_id'] == player_id]
    if vals.empty:
        return None
    latest = vals.sort_values('datetime').iloc[-1]
    return latest['market_value_in_eur']

def get_player_stats(name: str, players_df: pd.DataFrame, valuations_df: pd.DataFrame, events_df: pd.DataFrame, appearances_df: pd.DataFrame) -> Dict[str, Any] or None:
    name_lower = name.strip().lower()
    row = players_df[players_df['name_lower'] == name_lower]
    if row.empty:
        row = players_df[players_df['name_lower'].str.contains(name_lower)]
        if row.empty:
            return None
    row = row.iloc[0]
    player_id = row['player_id']

    # Age
    try:
        dob = pd.to_datetime(row['date_of_birth'], errors='coerce')
        age = int((pd.Timestamp('today') - dob).days // 365.25) if pd.notnull(dob) else None
    except Exception:
        age = None

    # Market value
    market_value = get_latest_market_value(player_id, valuations_df)
    if market_value is None or pd.isna(market_value):
        market_value = row.get('market_value_in_eur', None)

    # Appearances, assists, minutes played
    assists = None
    minutes_played = None
    appearances = None
    if appearances_df is not None:
        player_apps = appearances_df[appearances_df['player_id'] == player_id]
        appearances = len(player_apps)
        assists = int(player_apps['assists'].sum()) if 'assists' in player_apps and not player_apps.empty else None
        minutes_played = int(player_apps['minutes_played'].sum()) if 'minutes_played' in player_apps and not player_apps.empty else None

    # Goals from events
    goals = int(events_df[(events_df['player_id'] == player_id) & (events_df['type'] == 'Goals')].shape[0])

    # Fallback for appearances if not available
    if appearances is None:
        appearances = len(events_df[(events_df['player_id'] == player_id)]['game_id'].unique())

    return {
        'age': age,
        'position': row['position'],
        'current_club': row.get('current_club_name', ''),
        'league': row.get('current_club_domestic_competition_id', ''),
        'appearances': appearances,
        'goals': goals,
        'assists': assists,
        'minutes_played': minutes_played,
        'market_value': market_value
    }

@st.cache_resource
def load_model():
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
    position_encoding = {'Forward': 3, 'Midfielder': 2, 'Defender': 1, 'Goalkeeper': 0}
    league_encoding = {'GB1': 4, 'ES1': 3, 'IT1': 2, 'L1': 1, 'FR1': 0}
    features = np.array([[
        stats['age'] if stats['age'] is not None else 0,
        position_encoding.get(stats['position'], 1),
        league_encoding.get(stats['league'], 1),
        stats['appearances'] if stats['appearances'] is not None else 0,
        stats['goals'] if stats['goals'] is not None else 0,
        stats['assists'] if stats['assists'] is not None else 0,
        stats['minutes_played'] if stats['minutes_played'] is not None else 0
    ]])
    prediction = model.predict(features)[0]
    return max(prediction, 0)

def display_feature_importance(model):
    feature_names = ['Age', 'Position', 'League', 'Appearances', 'Goals', 'Assists', 'Minutes Played']
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, importance, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance in Market Value Prediction')
    ax.grid(axis='x', alpha=0.3)
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{imp:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def main():
    st.title("⚽ Football Player Market Value Predictor")
    st.markdown("Enter a player's name to predict their market value using machine learning!")

    model = load_model()
    players_df = load_players()
    valuations_df = load_valuations()
    events_df = load_game_events()
    appearances_df = load_appearances()

    col1, col2 = st.columns([2, 1])
    with col1:
        player_name = st.text_input(
            "Enter Player Name:",
            placeholder="e.g., Lionel Messi, Eden Hazard, Cristiano Ronaldo"
        )
    with col2:
        predict_button = st.button("🔮 Predict Market Value", type="primary")

    if player_name and predict_button:
        with st.spinner("Searching for player in database..."):
            stats = get_player_stats(player_name, players_df, valuations_df, events_df, appearances_df)
        if stats is None:
            st.error(f"Player '{player_name}' not found in the database. Please try a different name or check the spelling.")
            return

        st.subheader(f"📊 Player Profile: {player_name}")
        stats_df = pd.DataFrame([
            ["Age", str(stats['age']) if stats['age'] is not None else "N/A"],
            ["Position", str(stats['position'])],
            ["Current Club", str(stats['current_club'])],
            ["League", str(stats['league'])],
            ["Appearances", str(stats['appearances']) if stats['appearances'] is not None else "N/A"],
            ["Goals", str(stats['goals']) if stats['goals'] is not None else "N/A"],
            ["Assists", str(stats['assists']) if stats['assists'] is not None else "N/A"],
            ["Minutes Played", str(stats['minutes_played']) if stats['minutes_played'] is not None else "N/A"],
            ["Latest Market Value", f"€{int(stats['market_value']):,}" if stats['market_value'] is not None else "N/A"]
        ], columns=["Statistic", "Value"])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.table(stats_df)
        with col2:
            predicted_value = predict_market_value(stats, model)
            st.metric(
                "💰 Predicted Market Value",
                f"€{predicted_value:,.0f}",
                help="Predicted market value in Euros"
            )
            if predicted_value > 100_000_000:
                st.success("🌟 World-class player!")
            elif predicted_value > 50_000_000:
                st.info("⭐ Elite player!")
            elif predicted_value > 20_000_000:
                st.info("🔥 High-value player!")
            else:
                st.info("📈 Developing player!")

        st.subheader("📈 Model Feature Importance")
        st.markdown("This chart shows which statistics most influence the market value prediction:")
        fig = display_feature_importance(model)
        st.pyplot(fig)

        with st.expander("ℹ️ Model Information"):
            st.write(f"**Model Type:** Random Forest Regressor")
            st.write(f"**Number of Features:** {len(model.feature_importances_)}")
            st.write(f"**Training Data:** Synthetic dataset from Top 5 European leagues")
            st.write("**Features Used:** Age, Position, League, Appearances, Goals, Assists, Minutes Played")

    elif player_name and not predict_button:
        st.info("👆 Click the 'Predict Market Value' button to see the prediction!")

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app predicts football player market values using a Random Forest machine learning model.

        **How it works:**
        1. Enter any player name
        2. The app searches the database for real statistics
        3. ML model predicts market value
        4. feature importance analysis compares the players' stats to the rest of europe
        5. Try entering the full names for more accurate results
        6. Stats are updated as of October 2023

        **Leagues covered:**
        - Premier League 🏴
        - La Liga 🇪🇸
        - Serie A 🇮🇹
        - Bundesliga 🇩🇪
        - Ligue 1 🇫🇷
        """)

        st.header("Try these players:")
        example_players = [
            "Eden Hazard",
            "Lionel Messi",
            "Paulo Dybala"
        ]
        for player in example_players:
            if st.button(player, key=f"example_{player}"):
                st.session_state.player_name = player
                st.rerun()

if __name__ == "__main__":
    main()