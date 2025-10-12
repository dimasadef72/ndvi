import exifread

# Nama file foto
#foto = "D:\Server\Drone\Data ke-1\DJI_202409071655_007_sawah1\DJI_20240907165749_0001_D.JPG"
#foto = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data/DJI_20240907165749_0001_MS_NIR.TIF" 
foto = "/home/adedi/Documents/Tugas_Akhir/Computer Vision/NDVI/ndvi_results/DJI_20240907165749_0001_MS_NDVI.jpg"

# Baca file
with open(foto, 'rb') as f:
    tags = exifread.process_file(f)

print(f"=== METADATA {foto} ===\n")

for tag in tags.keys():
    # Skip JPEGThumbnail dan semua MakerNote
    if tag == 'JPEGThumbnail' or tag.startswith('MakerNote'):
        continue
    
    print(f"{tag}: {tags[tag]}")