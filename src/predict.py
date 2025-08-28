import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def create_sample_session_features(is_reel):
    if is_reel:
        session_features = {
            'total_volume_bytes': np.random.uniform(250000, 400000),
            'avg_packet_size': np.random.uniform(1250, 1400),
            'std_packet_size': np.random.uniform(100, 200),
            'session_duration': np.random.uniform(5, 15),
            'avg_inter_arrival_time': np.random.uniform(0.05, 0.15),
            'std_inter_arrival_time': np.random.uniform(0.2, 0.4),
            'packet_count': np.random.uniform(200, 300)
        }
    else:
        session_features = {
            'total_volume_bytes': np.random.uniform(40000, 80000),
            'avg_packet_size': np.random.uniform(300, 600),
            'std_packet_size': np.random.uniform(200, 350),
            'session_duration': np.random.uniform(10, 20),
            'avg_inter_arrival_time': np.random.uniform(0.1, 0.2),
            'std_inter_arrival_time': np.random.uniform(0.1, 0.25),
            'packet_count': np.random.uniform(80, 150)
        }
    return pd.DataFrame([session_features])

def main():
    # Define paths relative to the script's location
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / 'model' / 'traffic_classifier.joblib'
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}. Please run train.py first.")
        return

    sample_reel = create_sample_session_features(is_reel=True)
    reel_pred = model.predict(sample_reel)[0]
    reel_proba = model.predict_proba(sample_reel)[0]

    sample_non_reel = create_sample_session_features(is_reel=False)
    non_reel_pred = model.predict(sample_non_reel)[0]
    non_reel_proba = model.predict_proba(sample_non_reel)[0]
    
    print("\n==================================================")
    print("  🤖 Real-time Traffic Classification Report  ")
    print("==================================================")

    print("\n▶️  Analyzing Sample 1 (Simulated Reel Traffic)...")
    print(f"   - Predicted Class: {'Reel' if reel_pred == 1 else 'Non-Reel'}")
    print(f"   - Confidence -> [Non-Reel: {reel_proba[0]:.2%}, Reel: {reel_proba[1]:.2%}]")

    print("\n▶️  Analyzing Sample 2 (Simulated Non-Reel Traffic)...")
    print(f"   - Predicted Class: {'Reel' if non_reel_pred == 1 else 'Non-Reel'}")
    print(f"   - Confidence -> [Non-Reel: {non_reel_proba[0]:.2%}, Reel: {non_reel_proba[1]:.2%}]")
    print("\n==================================================")

if __name__ == "__main__":
    main()