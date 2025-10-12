import cv2
import numpy as np
import matplotlib
from PIL import Image
import piexif
import os

# ===== KONFIGURASI PATH =====
nir_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data"
red_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data"
output_folder = "./ndvi_results"

# ===== BUAT FOLDER OUTPUT =====
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

# ===== SCAN FILE NIR =====
print("\n=== Scanning NIR files ===")
nir_files = sorted([f for f in os.listdir(nir_folder) if f.endswith('_NIR.TIF')])
print(f"Found {len(nir_files)} NIR files")

if len(nir_files) == 0:
    print("ERROR: No NIR files found! Check your nir_folder path.")
    exit(1)

# ===== SCAN FILE RED =====
print("\n=== Scanning Red files ===")
red_files = sorted([f for f in os.listdir(red_folder) if f.endswith('_R.TIF')])
print(f"Found {len(red_files)} Red files")

if len(red_files) == 0:
    print("ERROR: No Red files found! Check your red_folder path.")
    exit(1)

# ===== PROSES KONVERSI =====
print(f"\n=== Starting NDVI conversion for {len(nir_files)} pairs ===\n")

success_count = 0
error_count = 0
colormap = matplotlib.colormaps['RdYlGn']

for i, nir_file in enumerate(nir_files, 1):
    # Extract base name untuk matching
    base_name = nir_file.replace('_NIR.TIF', '')
    red_file = f'{base_name}_R.TIF'

    # Cek apakah pasangan Red ada
    if red_file not in red_files:
        print(f"[{i}/{len(nir_files)}] ⚠ SKIP: No matching Red file for {nir_file}")
        error_count += 1
        continue

    # Path lengkap
    nir_path = os.path.join(nir_folder, nir_file)
    red_path = os.path.join(red_folder, red_file)
    output_path = os.path.join(output_folder, f'{base_name}_NDVI.jpg')

    try:
        # Baca gambar
        nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        red_image = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)

        if nir_image is None or red_image is None:
            print(f"[{i}/{len(nir_files)}] ✗ ERROR: Cannot read {base_name}")
            error_count += 1
            continue

        # Convert ke float
        nir = nir_image.astype(float)
        red = red_image.astype(float)

        # Hitung NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Normalisasi ke 0-255
        ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colormap RdYlGn
        ndvi_colored = (colormap(ndvi_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)

        # Convert RGB to BGR
        ndvi_bgr = cv2.cvtColor(ndvi_colored, cv2.COLOR_RGB2BGR)

        # Simpan gambar
        cv2.imwrite(output_path, ndvi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Copy EXIF dari NIR
        try:
            exif_dict = piexif.load(nir_path)
            ndvi_image_pil = Image.open(output_path)
            exif_bytes = piexif.dump(exif_dict)
            ndvi_image_pil.save(output_path, "jpeg", exif=exif_bytes)
        except:
            pass  # Skip jika EXIF error, tetap lanjut

        print(f"[{i}/{len(nir_files)}] ✓ {base_name}_NDVI.jpg (NDVI: {ndvi.min():.3f} ~ {ndvi.max():.3f})")
        success_count += 1

    except Exception as e:
        print(f"[{i}/{len(nir_files)}] ✗ ERROR: {base_name} - {e}")
        error_count += 1

# ===== SUMMARY =====
print("\n" + "="*60)
print("=== CONVERSION SUMMARY ===")
print(f"Total NIR files: {len(nir_files)}")
print(f"✓ Success: {success_count}")
print(f"✗ Failed: {error_count}")
print(f"Output folder: {output_folder}")
print("="*60)

if success_count > 0:
    print(f"\n✓ {success_count} NDVI images successfully created!")
    print(f"   Location: {os.path.abspath(output_folder)}")
    print("\nColor interpretation:")
    print("  🔴 Red    = Low vegetation (unhealthy/bare soil)")
    print("  🟡 Yellow = Moderate vegetation")
    print("  🟢 Green  = High vegetation (healthy plants)")
else:
    print("\n✗ No NDVI images created. Check your input folders and file names.")
