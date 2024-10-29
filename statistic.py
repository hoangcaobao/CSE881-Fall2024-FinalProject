import pickle

labels = ["Stop Sign", "Yield Sign", "Speed Limit Sign", "Pedestrian Crossing Sign", "No Entry Sign"]

with open("image_urls_all.pkl", "rb") as f:
    urls = pickle.load(f)

for i in range(len(labels)):
    print(f"{labels[i]} has {len(urls[i])} images")
