import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
import piexif

# Path input
nir_path = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data/DJI_20240907165749_0001_MS_NIR.TIF"
red_path = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data/DJI_20240907165749_0001_MS_R.TIF"

# Path output
output_path = "./DJI_20240907165749_0001_MS_NDVI.jpg"

# Baca gambar NIR dan Red sebagai grayscale
print("Reading NIR image...")
nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)

print("Reading Red image...")
red_image = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)

if nir_image is None:
    print(f"Error: Cannot read NIR image at {nir_path}")
    exit(1)

if red_image is None:
    print(f"Error: Cannot read Red image at {red_path}")
    exit(1)

print(f"NIR image shape: {nir_image.shape}, dtype: {nir_image.dtype}")
print(f"Red image shape: {red_image.shape}, dtype: {red_image.dtype}")

# Convert ke float untuk perhitungan
print("Calculating NDVI...")
nir = nir_image.astype(float)
red = red_image.astype(float)

# Hitung NDVI: (NIR - Red) / (NIR + Red)
ndvi = (nir - red) / (nir + red + 1e-10)

print(f"NDVI range: min={ndvi.min():.3f}, max={ndvi.max():.3f}")

# Normalisasi ke 0-255
ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply colormap RdYlGn (Red-Yellow-Green)
print("Applying colormap...")
import matplotlib
colormap = matplotlib.colormaps['RdYlGn']
ndvi_colored = (colormap(ndvi_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)

# Convert RGB to BGR untuk OpenCV
ndvi_bgr = cv2.cvtColor(ndvi_colored, cv2.COLOR_RGB2BGR)

# Simpan gambar
print(f"Saving NDVI image to {output_path}...")
cv2.imwrite(output_path, ndvi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# Copy EXIF dari NIR image
try:
    print("Copying EXIF metadata...")
    exif_dict = piexif.load(nir_path)
    ndvi_image_pil = Image.open(output_path)
    exif_bytes = piexif.dump(exif_dict)
    ndvi_image_pil.save(output_path, "jpeg", exif=exif_bytes)
    print("EXIF metadata copied successfully")
except Exception as e:
    print(f"Warning: Could not copy EXIF metadata: {e}")

print(f"\n✓ NDVI image successfully created: {output_path}")
print(f"  - Red areas: Low vegetation (unhealthy/bare soil)")
print(f"  - Yellow areas: Moderate vegetation")
print(f"  - Green areas: High vegetation (healthy plants)")
