import rasterio
import numpy as np

# Path ke file orthomosaic
nir_file = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data/DJI_20240907165749_0001_MS_NIR.TIF"
red_file = "/home/adedi/Documents/Tugas_Akhir/Data/File Agisoft/red_fix.tif"

# Cek file NIR
print("=== NIR File ===")
with rasterio.open(nir_file) as src:
    print(f"Jumlah band: {src.count}")
    print(f"Dimensi: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Data type: {src.dtypes}")

    for i in range(1, src.count + 1):
        band_data = src.read(i)
        print(f"\nBand {i}:")
        print(f"  Range: {band_data.min()} - {band_data.max()}")
        print(f"  Mean: {band_data.mean():.2f}")
        print(f"  Std: {band_data.std():.2f}")

        # Hitung persentase nilai unik
        unique = np.unique(band_data)
        print(f"  Unique values: {len(unique)}")
        if len(unique) <= 10:
            print(f"  Unique value list: {unique}")

        # Hitung histogram untuk 5 nilai paling umum
        vals, counts = np.unique(band_data, return_counts=True)
        top_5_idx = np.argsort(counts)[-5:][::-1]
        print(f"  Top 5 most common values:")
        for idx in top_5_idx:
            pct = counts[idx] / band_data.size * 100
            print(f"    {vals[idx]}: {counts[idx]} pixels ({pct:.2f}%)")

print("\n=== Red File ===")
with rasterio.open(red_file) as src:
    print(f"Jumlah band: {src.count}")
    print(f"Dimensi: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Data type: {src.dtypes}")

    for i in range(1, src.count + 1):
        band_data = src.read(i)
        print(f"\nBand {i}:")
        print(f"  Range: {band_data.min()} - {band_data.max()}")
        print(f"  Mean: {band_data.mean():.2f}")
        print(f"  Std: {band_data.std():.2f}")

        # Hitung persentase nilai unik
        unique = np.unique(band_data)
        print(f"  Unique values: {len(unique)}")
        if len(unique) <= 10:
            print(f"  Unique value list: {unique}")

        # Hitung histogram untuk 5 nilai paling umum
        vals, counts = np.unique(band_data, return_counts=True)
        top_5_idx = np.argsort(counts)[-5:][::-1]
        print(f"  Top 5 most common values:")
        for idx in top_5_idx:
            pct = counts[idx] / band_data.size * 100
            print(f"    {vals[idx]}: {counts[idx]} pixels ({pct:.2f}%)")

