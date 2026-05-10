import splitfolders
import os

# Definisi path direktori
input_dir = os.path.join("data", "raw", "Pokemon TCG")
output_dir = os.path.join("data", "processed")

print("Memulai proses pembagian dataset...")

# Proses pembagian data dengan rasio 70:15:15
splitfolders.ratio(
    input_dir, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None
)

print(f"Dataset berhasil dibagi dan disimpan di direktori: {output_dir}")
print("- Folder 'train' (70%)")
print("- Folder 'val' (15%)")
print("- Folder 'test' (15%)")
