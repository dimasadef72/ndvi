import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from rasterio import features
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import geopandas as gpd

def highlight_spray_zones_only(input_tif, output_png, output_tif, threshold_min=0.0, threshold_max=0.6):
    """
    OUTPUT 2: Highlight HANYA area NDVI 0-0.6 dengan warna (merah = butuh semprot)
    Selain itu TRANSPARAN.

    Parameters:
    - input_tif: path to input NDVI grayscale .tif
    - output_png: path to output PNG dengan transparansi
    - output_tif: path to output GeoTIFF dengan transparansi
    - threshold_min: NDVI minimum untuk highlight (default: 0.0)
    - threshold_max: NDVI maximum untuk highlight (default: 0.6)
    """

    print(f"\n{'='*80}")
    print(f"OUTPUT 2: SPRAY ZONES ONLY - NDVI {threshold_min} to {threshold_max}")
    print(f"{'='*80}\n")
    print(f"Input:  {input_tif}")
    print(f"Output PNG: {output_png}")
    print(f"Output TIF: {output_tif}\n")

    with rasterio.open(input_tif) as src:
        ndvi = src.read(1)
        profile = src.profile.copy()
        mask = src.read_masks(1)
        valid_mask = mask > 0

        print(f"NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
        print(f"Valid pixels: {np.sum(valid_mask):,}")

        # Handle NaN
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
        ndvi_clipped = np.clip(ndvi_clean, -1, 1)

        # Inisialisasi RGBA (semua transparan)
        height, width = ndvi_clipped.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Mask untuk spray zone (NDVI 0-0.6)
        spray_mask = (ndvi_clipped >= threshold_min) & (ndvi_clipped <= threshold_max) & valid_mask

        print(f"\nArea spray zone (NDVI {threshold_min}-{threshold_max}):")
        print(f"  Pixels: {np.sum(spray_mask):,}")
        if np.sum(valid_mask) > 0:
            print(f"  Percentage: {(np.sum(spray_mask) / np.sum(valid_mask) * 100):.2f}%")

        # Map NDVI 0-0.6 ke colormap RdYlGn (0.0-0.3 = merah ke oranye)
        # NDVI lebih rendah = lebih merah (prioritas semprot lebih tinggi)
        ndvi_for_cmap = np.zeros_like(ndvi_clipped, dtype=float)
        ndvi_for_cmap[spray_mask] = ndvi_clipped[spray_mask] / threshold_max * 0.3

        # Apply colormap RdYlGn
        cmap = plt.cm.RdYlGn
        rgba_colored = (cmap(ndvi_for_cmap) * 255).astype(np.uint8)

        # Set warna hanya untuk spray zone
        rgba[spray_mask] = rgba_colored[spray_mask]
        rgba[spray_mask, 3] = 255  # Opaque untuk area spray

        # Area lain tetap transparan (alpha = 0)

        # === Simpan sebagai PNG ===
        print(f"\nMenyimpan PNG: {output_png}")
        img = Image.fromarray(rgba, mode='RGBA')
        img.save(output_png, 'PNG')
        print(f"✓ PNG disimpan: {output_png}")

        # === Simpan sebagai GeoTIFF dengan transparansi ===
        print(f"\nMenyimpan GeoTIFF: {output_tif}")

        # Hapus file output dan aux file jika sudah ada
        if os.path.exists(output_tif):
            os.remove(output_tif)
        if os.path.exists(output_tif + '.aux.xml'):
            os.remove(output_tif + '.aux.xml')

        # Update profile untuk RGBA
        profile.update(
            dtype=rasterio.uint8,
            count=4,
            photometric='RGB',
            compress='lzw',
            nodata=None
        )

        with rasterio.open(output_tif, 'w', **profile) as dst:
            # Set band description untuk QGIS
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            dst.set_band_description(4, 'Alpha')

            dst.write(rgba[:, :, 0], 1)  # Red
            dst.write(rgba[:, :, 1], 2)  # Green
            dst.write(rgba[:, :, 2], 3)  # Blue
            dst.write(rgba[:, :, 3], 4)  # Alpha

            # Set mask untuk transparansi
            dst.write_mask(rgba[:, :, 3] > 0)

        print(f"✓ GeoTIFF disimpan: {output_tif}")
        print(f"\n{'='*80}")
        print("✓ SELESAI - OUTPUT 2")
        print(f"{'='*80}\n")

def pixels_to_polygons_merged(input_tif, output_shp, threshold_min=0.0, threshold_max=0.6, merge_distance=10):
    """
    Konversi pixel NDVI 0-0.6 menjadi polygon dan gabungkan berdasarkan jarak terdekat.

    Parameters:
    - input_tif: path to input NDVI grayscale .tif
    - output_shp: path to output shapefile
    - threshold_min: NDVI minimum untuk konversi (default: 0.0)
    - threshold_max: NDVI maximum untuk konversi (default: 0.6)
    - merge_distance: jarak maksimum (dalam meter) untuk menggabungkan polygon (default: 10m)
    """

    print(f"\n{'='*80}")
    print(f"KONVERSI PIXEL KE POLYGON - NDVI {threshold_min} to {threshold_max}")
    print(f"{'='*80}\n")
    print(f"Input:  {input_tif}")
    print(f"Output: {output_shp}")
    print(f"Merge distance: {merge_distance} meter\n")

    with rasterio.open(input_tif) as src:
        ndvi = src.read(1)
        transform = src.transform
        crs = src.crs

        # Handle NaN
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
        ndvi_clipped = np.clip(ndvi_clean, -1, 1)

        # Buat binary mask untuk spray zone (NDVI 0-0.6)
        mask = src.read_masks(1)
        valid_mask = mask > 0
        spray_mask = (ndvi_clipped >= threshold_min) & (ndvi_clipped <= threshold_max) & valid_mask

        # Konversi ke uint8 untuk rasterio.features.shapes
        spray_mask_uint8 = spray_mask.astype(np.uint8)

        print(f"Total pixels dalam spray zone: {np.sum(spray_mask):,}")

        # Ekstrak polygon dari raster
        print("\n[STEP 1] Ekstrak polygon dari pixel...")
        polygons = []
        ndvi_values = []

        for geom, value in features.shapes(spray_mask_uint8, transform=transform):
            if value == 1:  # Hanya ambil area spray zone
                poly = shape(geom)
                polygons.append(poly)
                # Ambil rata-rata NDVI untuk polygon ini
                # (untuk keperluan analisis nantinya)

        print(f"Total polygon awal: {len(polygons)}")

        if len(polygons) == 0:
            print("Tidak ada polygon yang ditemukan!")
            return

        # Buat GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)

        # Konversi ke UTM jika masih dalam Geographic CRS
        print(f"\n[STEP 2] Konversi ke projected CRS (UTM)...")
        original_crs = crs

        if crs.is_geographic:
            # Auto-detect UTM zone berdasarkan centroid
            # Format: EPSG:326XX untuk UTM North, EPSG:327XX untuk UTM South
            # XX = zone number (1-60)
            centroid = gdf.union_all().centroid
            lon = centroid.x
            lat = centroid.y

            # Hitung UTM zone
            utm_zone = int((lon + 180) / 6) + 1

            # Tentukan hemisphere (North/South)
            if lat >= 0:
                utm_epsg = 32600 + utm_zone  # UTM North
            else:
                utm_epsg = 32700 + utm_zone  # UTM South

            print(f"  CRS asli: {crs}")
            print(f"  Konversi ke: EPSG:{utm_epsg} (UTM Zone {utm_zone}{'N' if lat >= 0 else 'S'})")

            gdf = gdf.to_crs(epsg=utm_epsg)
        else:
            print(f"  CRS sudah projected: {crs}")

        # Buffer untuk merge polygon yang berdekatan
        print(f"\n[STEP 3] Gabungkan polygon dengan jarak <= {merge_distance}m...")

        # Metode dissolve berdasarkan spatial join + explode
        # Hanya merge polygon yang benar-benar berdekatan
        buffered = gdf.copy()
        buffered['geometry'] = gdf.geometry.buffer(merge_distance / 2)

        # Dissolve/union semua buffer
        dissolved = buffered.dissolve()

        # Explode untuk memisahkan MultiPolygon menjadi individual polygons
        exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

        # Hitung area SEBELUM buffer negatif (area asli dari dissolve)
        exploded['area_m2'] = exploded.geometry.area
        exploded['area_ha'] = exploded['area_m2'] / 10000

        # Buffer balik negatif untuk setiap cluster terpisah
        exploded['geometry'] = exploded.geometry.buffer(-merge_distance / 2)

        # Filter polygon yang hilang setelah buffer negatif (terlalu kecil)
        merged_gdf = exploded[~exploded.geometry.is_empty].copy().reset_index(drop=True)

        # Debug: cek jumlah polygon sebelum konversi balik
        print(f"  Jumlah polygon setelah merge (di UTM): {len(merged_gdf)}")

        # # Simplify geometry untuk smooth polygon
        # print(f"\n[STEP 3.5] Simplify geometry...")
        # merged_gdf['geometry'] = merged_gdf.geometry.simplify(tolerance=2, preserve_topology=True)
        # print(f"  Tolerance: 2.0 meter")

        # Remove holes/inner rings dari polygon
        print(f"\n[STEP 3.6] Remove holes dari polygon...")
        from shapely.geometry import Polygon as ShapelyPolygon

        def remove_holes(geom):
            if geom.geom_type == 'Polygon':
                return ShapelyPolygon(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                return type(geom)([ShapelyPolygon(p.exterior.coords) for p in geom.geoms])
            return geom

        merged_gdf['geometry'] = merged_gdf.geometry.apply(remove_holes)
        print(f"  Holes removed from all polygons")

        # Konversi balik ke CRS asli
        if original_crs.is_geographic:
            print(f"  Konversi balik ke CRS asli...")
            merged_gdf = merged_gdf.to_crs(original_crs)

        # Hitung statistik untuk polygon yang sudah digabung
        print(f"\n[STEP 4] Hitung statistik untuk polygon yang sudah digabung...")
        print(f"  Total polygon untuk dihitung: {len(merged_gdf)}")

        merged_mean_ndvi = []

        for idx, poly in merged_gdf.iterrows():
            # Buat mask untuk polygon ini
            poly_mask = features.rasterize(
                [(mapping(poly.geometry), 1)],
                out_shape=ndvi.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            # Ambil nilai NDVI di dalam polygon (hanya yang spray zone 0-0.6!)
            ndvi_in_poly = ndvi_clipped[(poly_mask == 1) & spray_mask]
            mean_val = np.mean(ndvi_in_poly) if len(ndvi_in_poly) > 0 else 0
            merged_mean_ndvi.append(mean_val)

        merged_gdf['mean_ndvi'] = merged_mean_ndvi
        merged_gdf['id'] = range(1, len(merged_gdf) + 1)

        # Simpan ke shapefile
        print(f"\n[STEP 5] Simpan ke shapefile...")
        merged_gdf.to_file(output_shp)

        print(f"\n{'='*80}")
        print(f"HASIL:")
        print(f"  Polygon awal: {len(polygons)}")
        print(f"  Polygon setelah merge: {len(merged_gdf)}")
        print(f"  Total area: {merged_gdf['area_ha'].sum():.4f} Ha")
        print(f"  Mean NDVI: {merged_gdf['mean_ndvi'].mean():.4f}")
        print(f"\n  Output: {output_shp}")
        print(f"{'='*80}\n")

        return merged_gdf

if __name__ == "__main__":
    # Contoh penggunaan
    input_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/rute_its_clipped_fixed_geser.tif"

    # OUTPUT 2: Hanya area spray (NDVI 0-0.6)
    output_spray_zones_png = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Rute Drone/rute_its_spray_zones.png"
    output_spray_zones_tif = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Rute Drone/rute_its_spray_zones.tif"
    output_spray_zones_shp = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Rute Drone/rute_its_spray_zones.shp"

    # Step 1: Buat spray zones TIF
    highlight_spray_zones_only(input_file, output_spray_zones_png, output_spray_zones_tif)

    # Step 2: Konversi ke polygon dan merge
    pixels_to_polygons_merged(input_file, output_spray_zones_shp, merge_distance=0.5)

