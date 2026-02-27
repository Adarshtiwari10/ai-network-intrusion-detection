import numpy as np
from datetime import datetime

def get_random_packet(X_test, y_test):
    idx = np.random.randint(0, len(X_test))
    return X_test.iloc[idx], y_test.iloc[idx]

def predict(model, packet):
    return model.predict([packet])[0]

def simulate_window(model, X_test, y_test, window_size=50, threshold=0.6):

    indices = np.random.choice(len(X_test), window_size, replace=False)

    X_window = X_test.iloc[indices]

    probabilities = model.predict_proba(X_window)[:, 1]
    mean_risk = float(np.mean(probabilities))

    predictions = (probabilities > 0.5).astype(int)
    attack_count = int(np.sum(predictions == 1))

    if mean_risk<0.4:
        severity = "LOW"
    elif mean_risk<0.6:
        severity = "MEDIUM"
    else:        
        severity = "HIGH"

    alert_triggered = bool(mean_risk > threshold)

    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "window_size": int(window_size),
        "attack_count": attack_count,
        "mean_risk_score": round(mean_risk, 2),
        "severity": severity,
        "alert_triggered": alert_triggered
    }

    return event
