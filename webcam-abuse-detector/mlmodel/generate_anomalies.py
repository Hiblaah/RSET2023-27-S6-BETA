import pandas as pd
import numpy as np

# number of suspicious samples to generate
N = 200

data = []

for i in range(N):

    sample = {
        "is_known_app": 0,                    # unknown app
        "is_foreground": np.random.choice([0,1]),
        "user_active": 0,                     # user inactive
        "is_night": np.random.choice([0,1]),
        "has_network_connection": 1,
        "network_connection_count": np.random.randint(3,15),
        "duration_minutes": np.random.uniform(15,60),
        "label": 1
    }

    data.append(sample)

df = pd.DataFrame(data)

df.to_excel("generated_suspicious_samples.xlsx", index=False)

print("Generated", len(df), "suspicious samples")