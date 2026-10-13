
import glob
import os
import pandas as pd

folder = "results/feature_group_experiments"
files = glob.glob(os.path.join(folder, "*.csv"))

for file in files:
    print("\n" + "=" * 70)
    print(os.path.basename(file))
    print("=" * 70)

    try:
        df = pd.read_csv(file)
        print(df.to_string(index=False))
    except Exception as error:
        print(f"Could not read this file: {error}")
