import cv2
import numpy as np
import matplotlib
from PIL import Image
import piexif
import os

# ===== KONFIGURASI PATH =====
nir_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data"
red_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data"
output_folder = "./ndvi_orthomosaic_output"
ndvi_temp_folder = os.path.join(output_folder, "ndvi_images")

# ===== PARAMETER =====
scale_percent = 30        # Resize ke 30% untuk stitching (hemat RAM)
jpeg_quality = 90         # Quality NDVI images

# ===== BUAT FOLDER =====
for folder in [output_folder, ndvi_temp_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("="*70)
print("NDVI ORTHOMOSAIC GENERATOR (Direct Stitching)")
print("="*70)

# ===== STEP 1: KONVERSI NIR + RED → NDVI =====
print("\n" + "="*70)
print("STEP 1: CONVERTING NIR + RED → NDVI")
print("="*70)

nir_files = sorted([f for f in os.listdir(nir_folder) if f.endswith('_NIR.TIF')])
red_files = sorted([f for f in os.listdir(red_folder) if f.endswith('_R.TIF')])

print(f"\nFound {len(nir_files)} NIR files")
print(f"Found {len(red_files)} Red files")

if len(nir_files) == 0:
    print("ERROR: No NIR files found!")
    exit(1)

success_count = 0
colormap = matplotlib.colormaps['RdYlGn']

for i, nir_file in enumerate(nir_files, 1):
    base_name = nir_file.replace('_NIR.TIF', '')
    red_file = f'{base_name}_R.TIF'

    if red_file not in red_files:
        print(f"[{i}/{len(nir_files)}] ⚠ SKIP: No Red file for {nir_file}")
        continue

    nir_path = os.path.join(nir_folder, nir_file)
    red_path = os.path.join(red_folder, red_file)
    output_path = os.path.join(ndvi_temp_folder, f'{base_name}_NDVI.jpg')

    try:
        # Baca gambar
        nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        red_image = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)

        if nir_image is None or red_image is None:
            continue

        # Hitung NDVI
        nir = nir_image.astype(float)
        red = red_image.astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Normalisasi & colormap
        ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ndvi_colored = (colormap(ndvi_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
        ndvi_bgr = cv2.cvtColor(ndvi_colored, cv2.COLOR_RGB2BGR)

        # Simpan
        cv2.imwrite(output_path, ndvi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        # Copy EXIF
        try:
            exif_dict = piexif.load(nir_path)
            ndvi_pil = Image.open(output_path)
            exif_bytes = piexif.dump(exif_dict)
            ndvi_pil.save(output_path, "jpeg", exif=exif_bytes)
        except:
            pass

        if i % 10 == 0 or i == len(nir_files):
            print(f"[{i}/{len(nir_files)}] Converted {i} images...")

        success_count += 1

    except Exception as e:
        print(f"[{i}/{len(nir_files)}] ✗ ERROR: {base_name} - {e}")

print(f"\n✓ Conversion complete: {success_count} NDVI images created")

if success_count < 2:
    print("✗ Need at least 2 NDVI images for stitching")
    exit(1)

# ===== STEP 2: LOAD & RESIZE IMAGES =====
print("\n" + "="*70)
print("STEP 2: LOADING NDVI IMAGES")
print("="*70)

ndvi_images = []
ndvi_filenames = sorted([f for f in os.listdir(ndvi_temp_folder) if f.endswith('.jpg')])

print(f"\nLoading {len(ndvi_filenames)} NDVI images (resize to {scale_percent}%)...")

for i, filename in enumerate(ndvi_filenames, 1):
    img_path = os.path.join(ndvi_temp_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Resize untuk hemat RAM
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    ndvi_images.append(resized)

    if i % 20 == 0 or i == len(ndvi_filenames):
        print(f"  Loaded {i}/{len(ndvi_filenames)} images...")

print(f"\n✓ Loaded {len(ndvi_images)} images for stitching")

# ===== STEP 3: STITCH ALL IMAGES → ORTHOMOSAIC =====
print("\n" + "="*70)
print(f"STEP 3: STITCHING {len(ndvi_images)} IMAGES → ORTHOMOSAIC")
print("="*70)
print("\n⚠ This may take 10-30 minutes depending on number of images...")
print("⚠ RAM usage may be high (4-8GB). Close other applications if needed.\n")

try:
    # Buat stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    print(f"Starting stitching {len(ndvi_images)} images...")
    print("  - Detecting features (SIFT/ORB)...")
    print("  - Matching features...")
    print("  - Estimating homography (RANSAC)...")
    print("  - Warping & blending...\n")

    # Stitch semua gambar sekaligus
    status, orthomosaic = stitcher.stitch(ndvi_images)

    if status == cv2.Stitcher_OK:
        # Simpan hasil
        output_path = os.path.join(output_folder, "NDVI_Orthomosaic.jpg")
        cv2.imwrite(output_path, orthomosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        print("="*70)
        print("✓ SUCCESS! ORTHOMOSAIC CREATED!")
        print("="*70)
        print(f"\nOrthomosaic size: {orthomosaic.shape[1]} x {orthomosaic.shape[0]} pixels")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"Location: {output_path}")
        print(f"\nNDVI images: {ndvi_temp_folder}")
        print("\nColor interpretation:")
        print("  🔴 Red    = Low NDVI (unhealthy/bare soil)")
        print("  🟡 Yellow = Moderate NDVI")
        print("  🟢 Green  = High NDVI (healthy plants)")
        print("="*70)

    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("✗ ERROR: Need more images or better overlap")
        print("  Solution: Increase overlap when capturing images")

    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("✗ ERROR: Homography estimation failed")
        print("  Possible causes:")
        print("  - Not enough feature matches (texture too homogeneous)")
        print("  - Overlap too low between images")
        print("  - Images too different (lighting, angle)")
        print("\n  Solutions:")
        print("  - Try increasing scale_percent (40-50%)")
        print("  - Check image overlap (should be 60-70%)")
        print("  - Use batch stitching instead (ndvitest.py)")

    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("✗ ERROR: Camera parameters adjustment failed")
        print("  Solution: Try batch stitching (ndvitest.py)")

    else:
        print(f"✗ ERROR: Stitching failed with status code: {status}")

except cv2.error as e:
    print(f"✗ OpenCV ERROR: {e}")
    print("\nThis usually happens when:")
    print("  - Too many images (>50-80)")
    print("  - RAM insufficient")
    print("  - Images have poor overlap")
    print("\nSolutions:")
    print("  1. Try reducing number of images (test with 20-30 first)")
    print("  2. Increase RAM or close other applications")
    print("  3. Use batch stitching (ndvitest.py with batch_size=20)")

except MemoryError:
    print("✗ MEMORY ERROR: Not enough RAM!")
    print("\nSolutions:")
    print("  1. Reduce scale_percent (try 20% instead of 30%)")
    print("  2. Close other applications")
    print("  3. Use batch stitching (ndvitest.py)")

print("\n" + "="*70)
