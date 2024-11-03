import pickle
from glob import glob


labels = ["Stop Sign", "Speed Limit Sign", "Crosswalk Sign", "Traffic Light"]
cnt = 0

for label in labels:
    paths = glob(f"images/{label}/*.jpg")
    cnt += len(paths)
    print(f"{label} has {len(paths)} images")

print(f"Total {cnt} images")