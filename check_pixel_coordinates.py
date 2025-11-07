import rasterio
import numpy as np
import json
import argparse

def get_pixel_coordinates(tif_path, output_json=None, max_pixels=100):
    """
    Get coordinates for pixels in a GeoTIFF file and save to JSON

    Args:
        tif_path: Path to the GeoTIFF file
        output_json: Path to output JSON file (default: same name as tif with .json extension)
        max_pixels: Maximum number of pixels to process (default: 100)
    """
    with rasterio.open(tif_path) as src:
        print(f"File: {tif_path}")
        print(f"Dimensions: {src.width} x {src.height} pixels")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print("-" * 60)

        height, width = src.shape
        total_pixels = height * width
        print(f"\nTotal pixels: {total_pixels:,}")
        print(f"Processing first {max_pixels} pixels...")

        # Prepare data structure
        pixel_data = {
            "metadata": {
                "file": tif_path,
                "width": width,
                "height": height,
                "crs": str(src.crs),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top
                },
                "transform": list(src.transform),
                "total_pixels": total_pixels,
                "processed_pixels": max_pixels
            },
            "pixels": []
        }

        # Generate coordinates for limited pixels
        count = 0
        for row in range(height):
            for col in range(width):
                if count >= max_pixels:
                    break

                # Get coordinate for this pixel
                x, y = rasterio.transform.xy(src.transform, row, col)

                pixel_data["pixels"].append({
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y
                })

                count += 1

            if count >= max_pixels:
                break

        # Determine output file name
        if output_json is None:
            output_json = tif_path.replace('.tif', '_coordinates.json')

        # Save to JSON
        print(f"\nSaving to {output_json}...")
        with open(output_json, 'w') as f:
            json.dump(pixel_data, f, indent=2)

        print(f"Done! Saved {len(pixel_data['pixels']):,} pixel coordinates to {output_json}")

        # Show file size
        import os
        file_size = os.path.getsize(output_json)
        print(f"File size: {file_size / (1024*1024):.2f} MB")


def main():
    # Set path file di sini
    tif_file = '/home/adedi/Documents/Tugas_Akhir/Data/Jember/Clipped/sawah3_clipped_noutm.tif'  # <-- Ubah path file di sini
    output_json = None  # Biarkan None untuk auto-generate nama, atau set custom seperti 'hasil.json'
    max_pixels = 100  # <-- Ubah jumlah piksel yang ingin di-cek

    get_pixel_coordinates(tif_file, output_json, max_pixels)


if __name__ == '__main__':
    main()
