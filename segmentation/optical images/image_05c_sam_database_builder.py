import pickle
import os
import pandas as pd
from numpy import sort

# DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam\crops"
# OUTPUT_PATH = DATA_PATH
# IMAGE_PATH = r"C:\tierspital\data raw\photos"

DATA_PATH = r"D:\My Drive\data processed\photos\segmentation sam"
OUTPUT_PATH = DATA_PATH

files = os.listdir(OUTPUT_PATH)
files = sort([x for x in files if x.endswith(".dat")])

rows = []
for file in files:
    print(f"Loading {file}")
    with open(rf"{OUTPUT_PATH}\{file}", "rb") as reader:
        row = pickle.load(reader)
    rows.append(row)
df = pd.DataFrame.from_dict(rows)
with open(rf"{OUTPUT_PATH}\data.dat", "wb") as writer:
    pickle.dump(df, writer)