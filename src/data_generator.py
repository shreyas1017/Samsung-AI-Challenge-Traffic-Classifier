import pandas as pd
import numpy as np
import time
from pathlib import Path

def generate_session_data(session_id, label):
    packets = []
    current_time = time.time()
    
    if label == 1:
        num_packets = np.random.randint(150, 300)
        for _ in range(num_packets):
            if np.random.rand() > 0.3:
                inter_arrival_time = np.random.uniform(0.001, 0.02)
            else:
                inter_arrival_time = np.random.uniform(0.5, 1.5)
            
            packet_size = np.random.normal(1300, 150)
            current_time += inter_arrival_time
            packets.append([session_id, current_time, max(100, packet_size), label])
    else:
        num_packets = np.random.randint(50, 150)
        for _ in range(num_packets):
            inter_arrival_time = np.random.exponential(0.1)
            packet_size = np.random.uniform(60, 800)
            current_time += inter_arrival_time
            packets.append([session_id, current_time, packet_size, label])
            
    return packets

def main():
    print("---  Generating Synthetic Network Traffic Data ---")
    all_packets = []
    num_sessions = 500

    for i in range(num_sessions):
        label = 1 if i % 2 == 0 else 0
        session_id = i
        all_packets.extend(generate_session_data(session_id, label))
        
    df = pd.DataFrame(all_packets, columns=['session_id', 'timestamp', 'packet_size', 'label'])
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Define paths relative to the script's location for robustness
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'data'
    output_dir.mkdir(exist_ok=True) # Create the 'data' directory if it doesn't exist
    output_path = output_dir / 'traffic_data_raw.csv'
    
    df.to_csv(output_path, index=False)
    print(f"✅ Data generated successfully and saved to {output_path}")

if __name__ == "__main__":
    main()