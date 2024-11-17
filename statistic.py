import pickle
from glob import glob
import pandas as pd

labels = ["Stop Sign", "Speed Limit Sign", "Crosswalk Sign", "Traffic Light"]
cnt = 0

for label in labels:
    paths = glob(f"images/{label}/*.jpg")
    cnt += len(paths)
    print(f"{label} has {len(paths)} images")

print(f"Total {cnt} images")

df = pd.read_csv("wrong_images_analysis/metadata_clip.csv")
error_type = df["error"].values
freq = {}
for error in error_type:
    if error not in freq:
        freq[error] = 0
    freq[error] += 1
print(freq)
