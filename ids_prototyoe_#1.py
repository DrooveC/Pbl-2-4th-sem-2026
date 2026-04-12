import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import random

# ==========================================
# 1. SYNTHETIC LOG GENERATION
# ==========================================
def generate_mock_logs(n_rows=1000):
    """
    Generates a dataset of 'Normal' web server logs and injects a few 'Anomalies'.
    """
    data = []
    start_time = datetime.now()
    
    # Normal behavior patterns
    ips = ["192.168.1.1", "192.168.1.2", "192.168.1.5", "10.0.0.15"]
    endpoints = ["/home", "/login", "/api/v1/data", "/contact", "/about"]
    methods = ["GET", "POST"]
    
    for i in range(n_rows):
        timestamp = start_time + timedelta(seconds=i * random.randint(1, 60))
        ip = random.choice(ips)
        endpoint = random.choice(endpoints)
        method = random.choice(methods)
        
        # Most requests are 200 OK, small response size
        status_code = 200 if random.random() > 0.05 else 404
        response_size = random.randint(200, 5000)
        
        data.append([timestamp, ip, method, endpoint, status_code, response_size])
    
    # Inject Anomalies (e.g., Data Exfiltration or Brute Force)
    # Anomaly 1: Massive response size (Data exfiltration)
    data.append([start_time, "99.99.99.99", "GET", "/api/v1/backup", 200, 5000000])
    
    # Anomaly 2: Rapid-fire 403 Forbidden errors (Scanner/Brute force)
    for i in range(10):
        data.append([start_time, "88.88.88.88", "POST", "/admin", 403, 150])

    return pd.DataFrame(data, columns=['timestamp', 'ip', 'method', 'endpoint', 'status', 'size'])

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def preprocess_logs(df):
    """
    Converts raw logs into numerical features for the ML model.
    """
    # Create a copy to avoid warnings
    features = pd.DataFrame()
    
    # Feature 1: Numerical Status Codes
    features['status'] = df['status']
    
    # Feature 2: Log of response size (to handle scaling)
    features['log_size'] = np.log1p(df['size'])
    
    # Feature 3: Request frequency (How many requests in that minute)
    df['minute'] = df['timestamp'].dt.floor('min')
    freq_map = df.groupby('minute').size().to_dict()
    features['freq'] = df['minute'].map(freq_map)
    
    # Feature 4: Encode Methods (Simplified)
    features['is_post'] = df['method'].apply(lambda x: 1 if x == 'POST' else 0)
    
    return features

# ==========================================
# 3. TRAINING & DETECTION
# ==========================================
def run_anomaly_detection():
    print("Step 1: Generating Logs...")
    raw_logs = generate_mock_logs(1000)
    
    print("Step 2: Engineering Features...")
    X = preprocess_logs(raw_logs)
    
    # We use Isolation Forest:
    # It works by randomly sub-sampling the data and building trees. 
    # Anomalies are easier to isolate and thus have shorter path lengths in the trees.
    model = IsolationForest(
        n_estimators=100,
        contamination=0.02, # We expect about 2% of data to be anomalous
        random_state=42
    )
    
    print("Step 3: Training Model...")
    # Predict returns -1 for anomalies and 1 for normal data
    raw_logs['anomaly_score'] = model.fit_predict(X)
    
    # Filter the results
    anomalies = raw_logs[raw_logs['anomaly_score'] == -1]
    
    print(f"\nDetection Complete. Found {len(anomalies)} anomalies.")
    print("-" * 50)
    print(anomalies[['timestamp', 'ip', 'endpoint', 'status', 'size']].head(10))
    print("-" * 50)

if __name__ == "__main__":
    run_anomaly_detection()
