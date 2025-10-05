import pandas as pd

# Load CSV
df = pd.read_csv("test.csv")

# Keep only these columns
columns_to_keep = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_srad', 'koi_smass', 'koi_impact', 'koi_teq',
    'koi_insol', 'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co']
df = df[columns_to_keep]

# Save back
df.to_csv("test_cleaned.csv", index=False)