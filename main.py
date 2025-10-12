#!/usr/bin/env python3
"""
NDVI Processing Pipeline - Step by Step
Step 1: Extract GPS Metadata dari foto drone
"""

import os
import exifread
from pathlib import Path


def dms_to_decimal(dms, ref):
    """Convert GPS coordinates from DMS to Decimal"""
    degrees = dms[0].num / dms[0].den
    minutes = dms[1].num / dms[1].den
    seconds = dms[2].num / dms[2].den

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if ref in ['S', 'W']:
        decimal = -decimal

    return decimal


def extract_gps_metadata(image_path):
    """Extract GPS dan metadata penting dari foto"""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    # Extract GPS
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat = dms_to_decimal(tags['GPS GPSLatitude'].values,
                            tags['GPS GPSLatitudeRef'].values)
        lon = dms_to_decimal(tags['GPS GPSLongitude'].values,
                            tags['GPS GPSLongitudeRef'].values)

        # Altitude
        alt = 0
        if 'GPS GPSAltitude' in tags:
            alt_value = tags['GPS GPSAltitude'].values[0]
            alt = alt_value.num / alt_value.den

        # Image dimensions
        width = int(tags.get('Image ImageWidth', 0).values[0])
        height = int(tags.get('Image ImageLength', 0).values[0])

        # Focal length
        focal = 0
        if 'EXIF FocalLength' in tags:
            focal_value = tags['EXIF FocalLength'].values[0]
            focal = focal_value.num / focal_value.den

        return {
            'path': str(image_path),
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'width': width,
            'height': height,
            'focal_length': focal
        }

    return None


def process_folder(folder_path):
    """Process semua foto di folder"""
    metadata_list = []

    folder = Path(folder_path)
    image_files = sorted(folder.glob('*.TIF')) + sorted(folder.glob('*.tif'))

    print(f"Found {len(image_files)} images in {folder_path}")

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        metadata = extract_gps_metadata(img_path)

        if metadata:
            metadata_list.append(metadata)
            print(f"  GPS: ({metadata['lat']:.6f}, {metadata['lon']:.6f})")
            print(f"  Alt: {metadata['alt']:.2f}m")
            print(f"  Size: {metadata['width']}x{metadata['height']}")
            print(f"  Focal: {metadata['focal_length']:.2f}mm")
        else:
            print(f"  WARNING: No GPS data found!")

    return metadata_list


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # TODO: Ganti path ini dengan lokasi folder foto NIR dan Red
    NIR_FOLDER = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data"
    RED_FOLDER = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data"

    print("=" * 60)
    print("STEP 1: EXTRACT GPS METADATA")
    print("=" * 60)

    # Process NIR images
    print("\n--- Processing NIR Images ---")
    nir_metadata = process_folder(NIR_FOLDER)

    print(f"\nTotal NIR images with GPS: {len(nir_metadata)}")

    if nir_metadata:
        print("\nSample NIR metadata (first image):")
        print(nir_metadata[0])

    # Uncomment untuk process RED images
    print("\n--- Processing RED Images ---")
    red_metadata = process_folder(RED_FOLDER)
    print(f"\nTotal RED images with GPS: {len(red_metadata)}")
