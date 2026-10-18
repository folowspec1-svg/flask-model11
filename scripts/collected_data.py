import csv
from pathlib import Path
from sklearn.datasets import load_diabetes

#load dataset
dataset = load_diabetes()

#Nama fitur
feature_names = dataset.feature_names

# Data Input (x) dan targrt ( y)
x = dataset.data
y = dataset.target

#Sace_file dataset ke foldet "data"
data_folder = Path("data")
data_file = data_folder / "diabetes_regression.csv"

# buka file CSV Dataset
with open(data_file, "w", newline="") as file:
    writer = csv.writer(file)

    header = list(feature_names) + ["target"]
    writer.writerow(header)

    for feature, target in zip(x,y):
        row = list(feature) + [target]
        writer.writerow(row)

#menampilkan informasi hasil
print("Dataset berhasil tersimpan")
print(f"Lokasi file: {data_file}")
print(f"jumlah data: {len(y)}")
print(f" Nama fitur: {','.join(feature_names)}")
