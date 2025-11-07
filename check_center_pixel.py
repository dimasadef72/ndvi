import rasterio

def check_pixel_at_coords(tif_path, pixels):
    """Check NDVI value at specific pixel coordinates

    Args:
        tif_path: Path to the GeoTIFF file
        pixels: List of [row, col] coordinates to check
    """
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        print(f"Image size: {width} x {height}")
        print("-" * 60)

        # Read NDVI band
        ndvi_band = src.read(1)

        # Check each pixel
        for row, col in pixels:
            if 0 <= row < height and 0 <= col < width:
                # Read NDVI value
                ndvi_value = float(ndvi_band[row, col])

                # Get coordinates
                lon, lat = rasterio.transform.xy(src.transform, row, col)

                print(f"\nPixel: [{row}, {col}]")
                print(f"Longitude: {lon:.6f}")
                print(f"Latitude: {lat:.6f}")
                print(f"NDVI value: {ndvi_value:.4f}")
            else:
                print(f"\nPixel: [{row}, {col}] - OUT OF BOUNDS!")


if __name__ == '__main__':
    tif_file = '/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/sawah3_clipped.tif'

    # Tentukan piksel yang ingin dicek [row, col]
    pixels_to_check = [
        [0, 0],           # Pojok kiri atas
        [100, 100],       # Sample
        [500, 500],       # Sample
        [1000, 1000],     # Sample
        [1595,2794],
        [2326,68],
    ]

    check_pixel_at_coords(tif_file, pixels_to_check)
