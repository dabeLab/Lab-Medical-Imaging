import datetime

import pandas as pd

# Load database
df = pd.read_csv("E:/2021_local_data/2023_Gd_synthesis/DICOM/database.csv", sep=";", index_col=False)

# Filter DataFrame
df = df[df["modality"].str.lower() == "mr"]
df = df.dropna(subset="series description")
df = df[(df["series description"].str.contains("t1", case=False,)) &
        (df["series description"].str.contains("tra", case=False,))]
df = df[df["patient species"] == "dog"]
df["interval"] = pd.Series(pd.Timedelta(0), dtype="timedelta64[ns]")
df["dataset"] = False

# Force datetime dtype for "series datetime" -> required to measure intervals
df["series datetime"] = pd.to_datetime(df["series datetime"], format="%Y.%m.%d %H:%M:%S")

grouped = df.groupby("patient id")
for patient, group in grouped:

    if len(group) == 3:

        group.sort_values("series datetime", inplace=True)
        group["interval"] = group["series datetime"].diff()

        # Assign the modified group back to the original DataFrame
        df.loc[group.index, 'interval'] = group['interval']

        # If Delta Time < 3 minutes, the time interval would not be enough for dose administration and imaging -> therefore the group must be discarded.
        if group["interval"].iloc[1:].gt(pd.Timedelta(minutes=3)).any():
            df.loc[group.index, "dataset"] = True

        # Add column with contrast dose
        df.loc[group.index, 'contrast dose'] = group["contrast dose"] = [0, 1 / 2, 1]


df = df[df["dataset"] == True].sort_values(["patient id", "series datetime"]).drop(columns="dataset")
df.to_csv("E:/2021_local_data/2023_Gd_synthesis/DICOM/database filtered.csv", sep=";", index=False)
print(df)







