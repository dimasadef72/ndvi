import rasterio
import numpy as np
import json

def check_pixel_ndvi(tif_path, output_json=None, specific_pixels=None):
    """
    Check coordinates and NDVI values for ALL pixels in a GeoTIFF file

    Args:
        tif_path: Path to the GeoTIFF file
        output_json: Path to output JSON file (default: auto-generated)
        specific_pixels: List of (row, col) tuples for specific pixels to check (optional)
    """
    with rasterio.open(tif_path) as src:
        print(f"File: {tif_path}")
        print(f"Dimensions: {src.width} x {src.height} pixels")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print("-" * 60)

        height, width = src.shape
        total_pixels = height * width

        # Read NDVI band
        ndvi_band = src.read(1)

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
                "total_pixels": total_pixels
            },
            "pixels": []
        }

        # Check specific pixels if provided
        if specific_pixels:
            print(f"\nChecking {len(specific_pixels)} specific pixels...")
            for row, col in specific_pixels:
                if 0 <= row < height and 0 <= col < width:
                    # Get coordinate for this pixel
                    x, y = rasterio.transform.xy(src.transform, row, col)

                    # Get NDVI value
                    ndvi_value = float(ndvi_band[row, col])

                    pixel_data["pixels"].append({
                        "row": row,
                        "col": col,
                        "longitude": x,
                        "latitude": y,
                        "ndvi": ndvi_value
                    })

                    print(f"  Pixel [{row}, {col}]: NDVI = {ndvi_value:.4f}, Lon/Lat = ({x:.6f}, {y:.6f})")
                else:
                    print(f"  Pixel [{row}, {col}]: OUT OF BOUNDS")
        else:
            # Process ALL pixels
            print(f"\nTotal pixels: {total_pixels:,}")
            print(f"Processing ALL pixels...")

            count = 0
            for row in range(height):
                for col in range(width):
                    # Get coordinate for this pixel
                    x, y = rasterio.transform.xy(src.transform, row, col)

                    # Get NDVI value
                    ndvi_value = float(ndvi_band[row, col])

                    pixel_data["pixels"].append({
                        "row": row,
                        "col": col,
                        "longitude": x,
                        "latitude": y,
                        "ndvi": ndvi_value
                    })

                    count += 1
                    if count % 10000 == 0:
                        print(f"  Processed {count:,} pixels...")

            pixel_data["metadata"]["processed_pixels"] = count

        # Determine output file name
        if output_json is None:
            output_json = tif_path.replace('.tif', '_pixel_ndvi.json')

        # Save to JSON
        print(f"\nSaving to {output_json}...")
        with open(output_json, 'w') as f:
            json.dump(pixel_data, f, indent=2)

        print(f"Done! Saved {len(pixel_data['pixels']):,} pixel data to {output_json}")

        # Show file size
        import os
        file_size = os.path.getsize(output_json)
        print(f"File size: {file_size / (1024*1024):.2f} MB")


def main():
    # Set path file di sini
    tif_file = '/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/sawah3_clipped.tif'
    output_json = None  # Biarkan None untuk auto-generate nama

    check_pixel_ndvi(tif_file, output_json)


if __name__ == '__main__':
    main()
