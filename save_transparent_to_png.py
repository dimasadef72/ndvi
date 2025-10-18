import cv2
import sys

def tif_to_png_with_transparency(input_tif, output_png=None):
    """
    Membaca file TIF dengan transparansi dan save ke PNG
    """
    print(f"Membaca file: {input_tif}")

    # Baca file dengan alpha channel
    img = cv2.imread(input_tif, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("❌ ERROR: Tidak dapat membaca file!")
        return

    print(f"✓ File berhasil dibaca")
    print(f"  Dimensi: {img.shape}")
    print(f"  Data type: {img.dtype}")

    # Tentukan output path jika tidak diberikan
    if output_png is None:
        output_png = input_tif.replace('.tif', '.png').replace('.TIF', '.png')

    # Simpan sebagai PNG
    cv2.imwrite(output_png, img)

    print(f"✓ File berhasil disimpan: {output_png}")

    # Info channels
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            print(f"  Format: RGBA (dengan transparansi)")
        else:
            print(f"  Format: RGB (tanpa transparansi)")
    else:
        print(f"  Format: Grayscale")

if __name__ == "__main__":
    # Default file
    default_file = "/home/adedi/Downloads/NDVI.tif"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_file
        print(f"Menggunakan file default: {input_file}")
        print(f"Atau jalankan: python save_transparent_to_png.py <path_to_file.tif>\n")

    # Optional output path
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    tif_to_png_with_transparency(input_file, output_file)
