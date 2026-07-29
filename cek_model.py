import joblib

artifact = joblib.load("model.pkl")

encoder = artifact["encoder"]
features = artifact["features"]

for fitur, kategori in zip(features, encoder.categories_):
    print("=" * 80)
    print(fitur)
    print(list(kategori))