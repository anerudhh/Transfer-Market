import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from data_loader import FootballDataLoader

def train_model_with_real_data():
    """
    Train a Random Forest model using real football data.
    """
    print("Loading real football dataset...")
    
    # Load data
    data_loader = FootballDataLoader()
    df = data_loader.get_top_5_leagues_players(limit=2000)
    
    if df.empty:
        raise ValueError("No training data available")
    
    print(f"Training dataset size: {len(df)} players")
    print(f"Market value range: €{df['market_value'].min():,.0f} - €{df['market_value'].max():,.0f}")
    print(f"Average market value: €{df['market_value'].mean():,.0f}")
    
    # Encode categorical variables
    position_encoder = LabelEncoder()
    league_encoder = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded['position_encoded'] = position_encoder.fit_transform(df['position'])
    df_encoded['league_encoded'] = league_encoder.fit_transform(df['league'])
    
    # Prepare features and target
    feature_columns = ['age', 'position_encoded', 'league_encoded', 'appearances', 'goals', 'assists', 'minutes_played']
    X = df_encoded[feature_columns]
    y = df_encoded['market_value']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
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
    feature_names = ['Age', 'Position', 'League', 'Appearances', 'Goals', 'Assists', 'Minutes Played']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Save model and encoders
    model_data = {
        'model': model,
        'position_encoder': position_encoder,
        'league_encoder': league_encoder,
        'feature_columns': feature_columns
    }
    
    with open('real_market_value_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel and encoders saved as 'real_market_value_model.pkl'")
    return model_data

if __name__ == "__main__":
    try:
        train_model_with_real_data()
        print("\nTraining completed successfully!")
        print("You can now run the updated Streamlit app with real player data.")
    except Exception as e:
        print(f"Error during training: {e}")