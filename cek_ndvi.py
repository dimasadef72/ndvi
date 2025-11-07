import cv2
import numpy as np
import matplotlib
import sys

def convert_ndvi_grayscale_to_color(file_path):
    """
    Convert NDVI grayscale TIF ke colormap custom (hitam-ungu-biru-hijau-kuning-merah)
    Range: -1 sampai 1
    """
    print(f"\n{'='*80}")
    print(f"CONVERT NDVI GRAYSCALE TO COLOR")
    print(f"{'='*80}\n")
    print(f"File: {file_path}\n")

    # Baca file sebagai float
    ndvi = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if ndvi is None:
        print("❌ ERROR: Tidak dapat membaca file!")
        return

    # Informasi dasar
    print("=== INFORMASI DASAR ===")
    print(f"Dimensi: {ndvi.shape}")
    print(f"Data type: {ndvi.dtype}")
    print(f"Min value: {ndvi.min():.6f}")
    print(f"Max value: {ndvi.max():.6f}")

    # Map NDVI dari range -1 sampai 1 ke range 0.0-1.0 untuk colormap
    # -1 -> 0.0 (merah), 0 -> 0.5 (kuning), 1 -> 1.0 (hijau)
    ndvi_scaled = (ndvi + 1.0) / 2.0
    ndvi_scaled = np.clip(ndvi_scaled, 0, 1)

    # Apply colormap RdYlGn
    colormap = matplotlib.colormaps['RdYlGn']
    ndvi_colored = (colormap(ndvi_scaled)[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR untuk OpenCV
    ndvi_colored_bgr = cv2.cvtColor(ndvi_colored, cv2.COLOR_RGB2BGR)

    # Simpan sebagai JPG
    output_file = file_path.replace('.TIF', '_colored.jpg').replace('.tif', '_colored.jpg')
    cv2.imwrite(output_file, ndvi_colored_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"\n✓ File berhasil diconvert dan disimpan: {output_file}")

    print(f"\n{'='*80}")
    print("✓ ANALISIS SELESAI")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Default file untuk testing
    default_file = "/home/adedi/Downloads/NDVI (1).tif"

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
        print(f"ℹ️  Menggunakan file default: {file_path}")
        print(f"   Atau jalankan: python cek_ndvi.py <path_to_ndvi.tif>\n")

    convert_ndvi_grayscale_to_color(file_path)
