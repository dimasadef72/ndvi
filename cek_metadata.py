import exifread
import subprocess
import json

# Nama file foto
#foto = "D:\Server\Drone\Data ke-1\DJI_202409071655_007_sawah1\DJI_20240907165749_0001_D.JPG"
foto_nir = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/rgb_data/DJI_20240907165749_0001_D.JPG"
foto_ndvi = "/home/adedi/Documents/Tugas_Akhir/Computer Vision/NDVI/ndvi_results/DJI_20240907165749_0001_MS_NDVI.jpg"

def cek_metadata_exif(foto):
    print(f"\n{'='*80}")
    print(f"=== EXIF METADATA: {foto} ===")
    print(f"{'='*80}\n")

    with open(foto, 'rb') as f:
        tags = exifread.process_file(f)

    for tag in tags.keys():
        if tag == 'JPEGThumbnail' or tag.startswith('MakerNote'):
            continue
        print(f"{tag}: {tags[tag]}")

def cek_metadata_xmp_dji(foto):
    print(f"\n{'='*80}")
    print(f"=== XMP METADATA DJI: {foto} ===")
    print(f"{'='*80}\n")

    try:
        # Gunakan exiftool untuk baca semua metadata DJI di XMP
        result = subprocess.run(
            ['exiftool', '-j', '-G', '-XMP-drone-dji:all', foto],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                xmp_data = data[0]
                dji_tags = {k: v for k, v in xmp_data.items() if 'drone-dji' in k.lower()}

                if dji_tags:
                    for key, value in sorted(dji_tags.items()):
                        print(f"{key}: {value}")
                else:
                    print("⚠️  Tidak ada XMP metadata DJI ditemukan")
            else:
                print("⚠️  Tidak ada metadata ditemukan")
        else:
            print(f"❌ Error running exiftool: {result.stderr}")
    except FileNotFoundError:
        print("❌ exiftool tidak terinstall. Install dengan: sudo apt install libimage-exiftool-perl")
    except Exception as e:
        print(f"❌ Error: {e}")

# Cek kedua file
print("\n" + "="*80)
print("PERBANDINGAN METADATA NIR vs NDVI")
print("="*80)

cek_metadata_exif(foto_nir)
cek_metadata_xmp_dji(foto_nir)

cek_metadata_exif(foto_ndvi)
cek_metadata_xmp_dji(foto_ndvi)