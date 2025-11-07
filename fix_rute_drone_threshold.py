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
        # print(f"\n[STEP 3.6] Remove holes dari polygon...")
        # from shapely.geometry import Polygon as ShapelyPolygon

        # def remove_holes(geom):
        #     if geom.geom_type == 'Polygon':
        #         return ShapelyPolygon(geom.exterior.coords)
        #     elif geom.geom_type == 'MultiPolygon':
        #         return type(geom)([ShapelyPolygon(p.exterior.coords) for p in geom.geoms])
        #     return geom

        # merged_gdf['geometry'] = merged_gdf.geometry.apply(remove_holes)
        # print(f"  Holes removed from all polygons")

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

def generate_drone_route(shapefile_path, output_route_shp, line_spacing=2.0):
    """
    Generate waypoints rute drone dengan pola zigzag di dalam polygon
    dan garis penghubung antar polygon.

    Parameters:
    - shapefile_path: path ke shapefile polygon spray zones
    - output_route_shp: path output untuk shapefile rute drone
    - line_spacing: jarak antar jalur zigzag dalam meter (default: 2m)
    """
    from shapely.geometry import LineString, Point, MultiLineString

    print(f"\n{'='*80}")
    print(f"GENERATE DRONE ROUTE")
    print(f"{'='*80}\n")
    print(f"Input shapefile: {shapefile_path}")
    print(f"Output route: {output_route_shp}")
    print(f"Line spacing: {line_spacing}m\n")

    # Baca shapefile polygon
    gdf = gpd.read_file(shapefile_path)

    # Fix invalid geometries
    gdf['geometry'] = gdf.geometry.buffer(0)

    # Konversi ke UTM jika belum
    original_crs = gdf.crs
    if gdf.crs.is_geographic:
        centroid = gdf.union_all().centroid
        lon = centroid.x
        lat = centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        gdf = gdf.to_crs(epsg=utm_epsg)
        print(f"Konversi ke UTM EPSG:{utm_epsg}")

    all_routes = []
    endpoints = []  # Untuk menyimpan titik awal dan akhir setiap polygon

    print(f"\n[STEP 1] Generate zigzag pattern untuk {len(gdf)} polygon...")

    for idx, row in gdf.iterrows():
        poly = row.geometry
        bounds = poly.bounds  # (minx, miny, maxx, maxy)

        # Tentukan arah zigzag (horizontal)
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny

        # Generate garis horizontal
        lines = []
        y = miny
        direction = 1  # 1 = kiri ke kanan, -1 = kanan ke kiri

        while y <= maxy:
            if direction == 1:
                line = LineString([(minx, y), (maxx, y)])
            else:
                line = LineString([(maxx, y), (minx, y)])

            # Potong garis dengan polygon
            clipped = line.intersection(poly)

            if not clipped.is_empty:
                if clipped.geom_type == 'LineString':
                    lines.append(clipped)
                elif clipped.geom_type == 'MultiLineString':
                    # Ambil segmen terpanjang jika ada multiple
                    longest = max(clipped.geoms, key=lambda x: x.length)
                    lines.append(longest)

            y += line_spacing
            direction *= -1  # Balik arah

        # Gabungkan semua garis zigzag dalam polygon ini
        if lines:
            # Hubungkan garis-garis menjadi satu rute kontinyu
            route_coords = []
            for i, line in enumerate(lines):
                coords = list(line.coords)
                if i == 0:
                    route_coords.extend(coords)
                else:
                    # Hubungkan ujung garis sebelumnya ke awal garis ini
                    route_coords.append(coords[0])  # Garis vertikal penghubung
                    route_coords.extend(coords)

            route = LineString(route_coords)
            all_routes.append(route)

            # Simpan endpoint untuk connecting antar polygon
            start_point = Point(route_coords[0])
            end_point = Point(route_coords[-1])
            endpoints.append({
                'id': idx,
                'start': start_point,
                'end': end_point,
                'route': route
            })

    print(f"  Total route segments: {len(all_routes)}")

    # [STEP 2] Hubungkan antar polygon dengan nearest neighbor
    print(f"\n[STEP 2] Hubungkan antar polygon (nearest neighbor)...")

    if len(endpoints) > 1:
        visited = [False] * len(endpoints)
        current_idx = 0  # Mulai dari polygon pertama
        visited[current_idx] = True
        ordered_routes = [endpoints[current_idx]['route']]
        current_point = endpoints[current_idx]['end']

        for _ in range(len(endpoints) - 1):
            # Cari polygon terdekat yang belum dikunjungi
            min_dist = float('inf')
            next_idx = -1
            connect_to_start = True

            for i, ep in enumerate(endpoints):
                if visited[i]:
                    continue

                # Cek jarak ke start dan end point
                dist_to_start = current_point.distance(ep['start'])
                dist_to_end = current_point.distance(ep['end'])

                if dist_to_start < min_dist:
                    min_dist = dist_to_start
                    next_idx = i
                    connect_to_start = True

                if dist_to_end < min_dist:
                    min_dist = dist_to_end
                    next_idx = i
                    connect_to_start = False

            if next_idx != -1:
                # Tambahkan garis penghubung
                next_ep = endpoints[next_idx]
                connection_point = next_ep['start'] if connect_to_start else next_ep['end']
                connecting_line = LineString([current_point.coords[0], connection_point.coords[0]])
                ordered_routes.append(connecting_line)

                # Tambahkan route polygon berikutnya (reverse jika perlu)
                next_route = next_ep['route']
                if not connect_to_start:
                    # Reverse route
                    next_route = LineString(list(next_route.coords)[::-1])

                ordered_routes.append(next_route)

                # Update current point dan visited
                visited[next_idx] = True
                if connect_to_start:
                    current_point = next_ep['end']
                else:
                    current_point = next_ep['start']

        all_routes = ordered_routes

    print(f"  Total segments (with connections): {len(all_routes)}")

    # Konversi balik ke CRS asli
    if original_crs.is_geographic:
        # Buat GeoDataFrame sementara untuk konversi
        temp_gdf = gpd.GeoDataFrame({'geometry': all_routes}, crs=gdf.crs)
        temp_gdf = temp_gdf.to_crs(original_crs)
        all_routes = list(temp_gdf.geometry)

    # Simpan sebagai shapefile
    print(f"\n[STEP 3] Simpan route ke shapefile...")
    route_gdf = gpd.GeoDataFrame({
        'segment_id': range(len(all_routes)),
        'geometry': all_routes
    }, crs=original_crs)

    route_gdf.to_file(output_route_shp)

    # Hitung total panjang rute
    total_length = sum([r.length for r in all_routes])

    print(f"\n{'='*80}")
    print(f"HASIL:")
    print(f"  Total segments: {len(all_routes)}")
    print(f"  Total panjang rute: {total_length:.2f} meter")
    print(f"  Output: {output_route_shp}")
    print(f"{'='*80}\n")

    return route_gdf

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

    # Step 3: Generate rute drone
    output_drone_route_shp = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Rute Drone/rute_its_drone_route.shp"
    generate_drone_route(output_spray_zones_shp, output_drone_route_shp, line_spacing=2.0)

