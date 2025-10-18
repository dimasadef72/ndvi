import cv2
import numpy as np
import os
import subprocess

# ===== KONFIGURASI PATH =====
nir_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data"
red_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data"
output_folder = "./ndvi_grayscale"

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
print(f"\n=== Starting NDVI conversion (Grayscale) for {len(nir_files)} pairs ===\n")

success_count = 0
error_count = 0

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
    output_path = os.path.join(output_folder, f'{base_name}_NDVI.TIF')

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

        # Hitung NDVI (range: -1 hingga +1)
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Simpan sebagai float32 TIFF (single band, grayscale)
        # NDVI disimpan dalam range aslinya (-1 to +1)
        ndvi_float32 = ndvi.astype(np.float32)
        cv2.imwrite(output_path, ndvi_float32)

        # Copy ALL metadata dari NIR menggunakan exiftool
        try:
            result = subprocess.run(
                ['exiftool', '-TagsFromFile', nir_path, '-all:all', '-overwrite_original', output_path],
                capture_output=True,
                text=True,
                check=False
            )
        except Exception as e:
            print(f"  ⚠ Warning: Could not copy metadata - {e}")

        print(f"[{i}/{len(nir_files)}] ✓ {base_name}_NDVI.TIF (NDVI: {ndvi.min():.3f} ~ {ndvi.max():.3f})")
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
    print(f"\n✓ {success_count} NDVI grayscale images successfully created!")
    print(f"   Location: {os.path.abspath(output_folder)}")
    print("\nFile format:")
    print("  • Single-band TIFF (Float32)")
    print("  • NDVI range: -1.0 to +1.0")
    print("  • Grayscale (no colormap)")
    print("  • All metadata preserved from NIR")
    print("\n📤 These files are ready for DJI SmartFarm upload!")
else:
    print("\n✗ No NDVI images created. Check your input folders and file names.")
