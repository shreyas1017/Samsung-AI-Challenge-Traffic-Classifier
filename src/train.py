import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import joblib
from pathlib import Path

def create_features(df):
    df_sorted = df.sort_values(by=['session_id', 'timestamp'])
    df_sorted['inter_arrival_time'] = df_sorted.groupby('session_id')['timestamp'].diff().fillna(0)

    features = df_sorted.groupby('session_id').agg(
        total_volume_bytes=('packet_size', 'sum'),
        avg_packet_size=('packet_size', 'mean'),
        std_packet_size=('packet_size', 'std'),
        session_duration=('timestamp', lambda x: x.max() - x.min()),
        avg_inter_arrival_time=('inter_arrival_time', 'mean'),
        std_inter_arrival_time=('inter_arrival_time', 'std'),
        packet_count=('packet_size', 'count')
    ).reset_index()

    labels = df.groupby('session_id')['label'].first().reset_index()
    
    final_df = pd.merge(features, labels, on='session_id')
    final_df = final_df.fillna(0)
    
    return final_df

def main():
    print("--- 🚀 Starting Model Training Pipeline ---")
    
    # Define paths relative to the script's location
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'traffic_data_raw.csv'
    model_dir = script_dir.parent / 'model'
    model_dir.mkdir(exist_ok=True) # Create the 'model' directory if it doesn't exist
    model_path = model_dir / 'traffic_classifier.joblib'

    try:
        print("\nStep 1/5: Loading raw traffic data...")
        raw_df = pd.read_csv(data_path)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Raw data not found at {data_path}. Please run data_generator.py first.")
        return

    print("\nStep 2/5: Engineering features from raw data...")
    featured_df = create_features(raw_df)
    print("✅ Feature engineering complete.")
    
    X = featured_df.drop(['session_id', 'label'], axis=1)
    y = featured_df['label']
    
    print("\nStep 3/5: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ Data split complete. Training set has {len(X_train)} samples.")
    
    print("\nStep 4/5: Training the LightGBM model...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    
    print("\nStep 5/5: Evaluating model and saving artifact...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Reel (Class 0)', 'Reel (Class 1)']))
    
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    print("\n--- 🎉 Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    main()