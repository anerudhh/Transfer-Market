import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import random

def generate_synthetic_dataset(n_players=500):
    """
    Generate a synthetic dataset of football players from top 5 European leagues.
    """
    print(f"Generating synthetic dataset with {n_players} players...")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define possible values
    positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    leagues = ['Premier League', 'La Liga', 'Serie A', 'Ligue 1', 'Bundesliga']
    
    # League multipliers for market value (Premier League typically has highest values)
    league_multipliers = {
        'Premier League': 1.3,
        'La Liga': 1.2,
        'Serie A': 1.1,
        'Bundesliga': 1.0,
        'Ligue 1': 0.9
    }
    
    # Position base values and goal/assist expectations
    position_config = {
        'Forward': {'base_value': 25_000_000, 'goal_weight': 2_000_000, 'assist_weight': 800_000},
        'Midfielder': {'base_value': 20_000_000, 'goal_weight': 1_500_000, 'assist_weight': 1_200_000},
        'Defender': {'base_value': 15_000_000, 'goal_weight': 3_000_000, 'assist_weight': 1_500_000},
        'Goalkeeper': {'base_value': 10_000_000, 'goal_weight': 5_000_000, 'assist_weight': 2_000_000}
    }
    
    data = []
    
    for i in range(n_players):
        # Basic attributes
        age = np.random.randint(18, 38)
        position = random.choice(positions)
        league = random.choice(leagues)
        
        # Performance stats based on position
        if position == 'Goalkeeper':
            appearances = np.random.randint(15, 38)
            goals = np.random.randint(0, 3)
            assists = np.random.randint(0, 4)
        elif position == 'Forward':
            appearances = np.random.randint(20, 38)
            goals = np.random.randint(5, 35)
            assists = np.random.randint(2, 15)
        elif position == 'Midfielder':
            appearances = np.random.randint(25, 38)
            goals = np.random.randint(2, 20)
            assists = np.random.randint(5, 20)
        else:  # Defender
            appearances = np.random.randint(25, 38)
            goals = np.random.randint(0, 8)
            assists = np.random.randint(1, 10)
        
        minutes_played = appearances * np.random.randint(70, 90)
        
        # Calculate market value based on multiple factors
        config = position_config[position]
        base_value = config['base_value']
        
        # Age factor (peak around 25-27)
        age_factor = 1.0
        if 23 <= age <= 28:
            age_factor = 1.2
        elif age < 21:
            age_factor = 0.7 + (age - 18) * 0.1  # Young potential
        elif age > 32:
            age_factor = 0.6
        
        # Performance factor
        performance_value = (goals * config['goal_weight'] + 
                           assists * config['assist_weight'] + 
                           appearances * 300_000)
        
        # League factor
        league_factor = league_multipliers[league]
        
        # Add some randomness
        randomness = np.random.normal(1.0, 0.3)
        randomness = max(0.3, min(2.0, randomness))  # Clamp between 0.3 and 2.0
        
        market_value = (base_value * age_factor + performance_value) * league_factor * randomness
        market_value = max(500_000, market_value)  # Minimum value
        
        # Encode categorical variables
        position_encoded = {'Forward': 3, 'Midfielder': 2, 'Defender': 1, 'Goalkeeper': 0}[position]
        league_encoded = {'Premier League': 4, 'La Liga': 3, 'Serie A': 2, 'Bundesliga': 1, 'Ligue 1': 0}[league]
        
        data.append({
            'age': age,
            'position': position_encoded,
            'league': league_encoded,
            'appearances': appearances,
            'goals': goals,
            'assists': assists,
            'minutes_played': minutes_played,
            'market_value': market_value
        })
    
    df = pd.DataFrame(data)
    print(f"Dataset generated successfully!")
    print(f"Market value range: €{df['market_value'].min():,.0f} - €{df['market_value'].max():,.0f}")
    print(f"Average market value: €{df['market_value'].mean():,.0f}")
    
    return df

def train_model(df):
    """
    Train a Random Forest model on the dataset.
    """
    print("Training Random Forest model...")
    
    # Prepare features and target
    feature_columns = ['age', 'position', 'league', 'appearances', 'goals', 'assists', 'minutes_played']
    X = df[feature_columns]
    y = df['market_value']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\nModel Performance:")
    print(f"Training MAE: €{train_mae:,.0f}")
    print(f"Testing MAE: €{test_mae:,.0f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Testing R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    return model

def save_model(model, filename='market_value_model.pkl'):
    """
    Save the trained model to a pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved as '{filename}'")

def main():
    print("Football Player Market Value Model Training")
    print("=" * 50)
    
    # Generate synthetic dataset
    df = generate_synthetic_dataset(n_players=500)
    
    # Train model
    model = train_model(df)
    
    # Save model
    save_model(model)
    
    print("\nTraining completed successfully!")
    print("You can now run 'streamlit run main.py' to use the web application.")

if __name__ == "__main__":
    main()
