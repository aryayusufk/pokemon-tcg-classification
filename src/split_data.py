import splitfolders
import os

# Definisi path direktori
input_dir = os.path.join("data", "raw", "Pokemon TCG")
output_dir = os.path.join("data", "processed (80-10-10)")

print("Memulai proses pembagian dataset...")

# Proses pembagian data dengan rasio 80:10:10
splitfolders.ratio(
    input_dir, output=output_dir, seed=42, ratio=(0.8, 0.1, 0.1), group_prefix=None
)

print(f"Dataset berhasil dibagi dan disimpan di direktori: {output_dir}")
print("- Folder 'train' (80%)")
print("- Folder 'val' (10%)")
print("- Folder 'test' (10%)")
