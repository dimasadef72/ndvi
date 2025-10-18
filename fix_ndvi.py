import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def apply_custom_ndvi_colormap(input_tif, output_png, output_tif, output_jpg=None):
    """
    Apply custom colormap to NDVI grayscale .tif (-1 to 1 range)
    Output: colored PNG, TIF with transparency preserved, and JPG visualization

    Klasifikasi:
    < 0: Hitam (Bukan Vegetasi)
    0 - 0.21: Merah (Tidak Sehat)
    0.21 - 0.4: Kuning (Kurang Sehat)
    0.4 - 0.6: Hijau Muda (Cukup Sehat)
    0.6 - 0.8: Hijau (Sehat)
    0.8 - 1.0: Hijau Tua (Sangat Sehat)

    Parameters:
    - input_tif: path to input NDVI grayscale .tif
    - output_png: path to output colored PNG
    - output_tif: path to output colored GeoTIFF
    - output_jpg: (optional) path to output visualization JPG with map + colorbar + stats
    """

    print(f"Membaca file: {input_tif}")

    # Baca file NDVI
    with rasterio.open(input_tif) as src:
        ndvi = src.read(1)
        profile = src.profile.copy()

        # Baca mask (area valid vs transparan)
        mask = src.read_masks(1)
        valid_mask = mask > 0

        print(f"NDVI range: {np.nanmin(ndvi):.3f} sampai {np.nanmax(ndvi):.3f}")
        print(f"Valid pixels: {np.sum(valid_mask)}, Transparent pixels: {np.sum(~valid_mask)}")

        # Debug: tampilkan sample nilai NDVI
        print(f"\nSample NDVI values (first 10x10):")
        print(ndvi[valid_mask][:100] if np.sum(valid_mask) >= 100 else ndvi[valid_mask])

        # Handle NaN dan invalid values
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip NDVI ke range [-1, 1]
        ndvi_clipped = np.clip(ndvi_clean, -1, 1)

        # Inisialisasi array RGBA
        height, width = ndvi_clipped.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Definisi warna untuk setiap kategori (RGB)
        # < 0: Hitam (Bukan Vegetasi)
        mask_black = (ndvi_clipped < 0) & valid_mask
        rgba[mask_black] = [0, 0, 0, 255]

        # >= 0: Gunakan RdYlGn colormap untuk semua nilai >= 0
        mask_vegetation = (ndvi_clipped >= 0) & valid_mask

        # Aplikasikan RdYlGn untuk nilai >= 0
        # Normalisasi dari range [0, 1.0] ke [0, 1] untuk colormap
        ndvi_for_cmap = np.zeros_like(ndvi_clipped, dtype=float)
        ndvi_for_cmap[mask_vegetation] = ndvi_clipped[mask_vegetation]  # Sudah 0-1

        cmap = plt.cm.RdYlGn
        rgba_colored = (cmap(ndvi_for_cmap) * 255).astype(np.uint8)

        # Terapkan warna untuk area >= 0
        rgba[mask_vegetation] = rgba_colored[mask_vegetation]

        # Set alpha channel ke 0 untuk area transparan
        rgba[~valid_mask, 3] = 0

        # Debug: cek berapa pixel per kategori sebelum save
        print(f"\nDebug - Pixel per kategori:")
        print(f"  Hitam (< 0): {np.sum(mask_black):,}")
        print(f"  RdYlGn (>= 0): {np.sum(mask_vegetation):,}")
        print(f"  Total colored: {np.sum([mask_black, mask_vegetation]):,}")

        # === Simpan sebagai PNG dengan transparansi ===
        print(f"\nMenyimpan PNG: {output_png}")
        img_png = Image.fromarray(rgba)  # Mode RGBA otomatis terdeteksi
        img_png.save(output_png, 'PNG')
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
            nodata=None  # Hapus nodata value karena pakai alpha channel
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

            # Set mask untuk transparansi - ini penting untuk QGIS
            dst.write_mask(rgba[:, :, 3] > 0)

        print(f"✓ GeoTIFF disimpan: {output_tif}")

        # === Hitung luas per pixel ===
        print(f"\n{'='*80}")
        print(f"DEBUG: PERHITUNGAN LUAS PER PIXEL")
        print(f"{'='*80}")

        pixel_width = abs(src.transform[0])
        pixel_height = abs(src.transform[4])

        print(f"\n[STEP 1] Ambil ukuran pixel dari GeoTIFF metadata:")
        print(f"  - src.transform[0] (pixel width)  = {src.transform[0]}")
        print(f"  - src.transform[4] (pixel height) = {src.transform[4]}")
        print(f"  - abs(pixel_width)  = {pixel_width}")
        print(f"  - abs(pixel_height) = {pixel_height}")

        # Cek apakah dalam degrees atau meter
        print(f"\n[STEP 2] Cek sistem koordinat:")
        print(f"  - CRS = {src.crs}")
        print(f"  - is_geographic? {src.crs.is_geographic if src.crs else 'No CRS'}")

        if src.crs and src.crs.is_geographic:
            import math
            bounds = src.bounds

            print(f"\n[STEP 3] Koordinat dalam DEGREES (WGS 84) - perlu konversi ke meter")
            print(f"  - Bounds: {bounds}")
            print(f"    * Top (utara)    = {bounds.top}°")
            print(f"    * Bottom (selatan) = {bounds.bottom}°")
            print(f"    * Left (barat)   = {bounds.left}°")
            print(f"    * Right (timur)  = {bounds.right}°")

            lat_center = (bounds.top + bounds.bottom) / 2
            print(f"\n[STEP 4] Hitung latitude tengah:")
            print(f"  - lat_center = (bounds.top + bounds.bottom) / 2")
            print(f"  - lat_center = ({bounds.top} + {bounds.bottom}) / 2")
            print(f"  - lat_center = {lat_center}°")

            print(f"\n[STEP 5] Konversi 1 degree ke meter:")
            print(f"  - Keliling bumi di ekuator ≈ 40,075 km")
            print(f"  - 1 degree = 40,075 / 360 = 111.32 km = 111,320 meter")

            lat_center_rad = math.radians(lat_center)
            cos_lat = math.cos(lat_center_rad)

            print(f"\n[STEP 5a] LATITUDE (utara-selatan):")
            print(f"  - meters_per_degree_lat = 111,320 (konstan)")
            meters_per_degree_lat = 111320

            print(f"\n[STEP 5b] LONGITUDE (timur-barat):")
            print(f"  - Konversi lat_center ke radian: {lat_center}° = {lat_center_rad:.6f} rad")
            print(f"  - cos(lat_center) = cos({lat_center_rad:.6f}) = {cos_lat:.6f}")
            print(f"  - meters_per_degree_lon = 111,320 × cos(lat_center)")
            print(f"  - meters_per_degree_lon = 111,320 × {cos_lat:.6f}")
            meters_per_degree_lon = 111320 * cos_lat
            print(f"  - meters_per_degree_lon = {meters_per_degree_lon:.4f} meter")

            print(f"\n[STEP 6] Konversi ukuran pixel dari degree ke meter:")
            print(f"  - pixel_width_m  = pixel_width × meters_per_degree_lon")
            print(f"  - pixel_width_m  = {pixel_width} × {meters_per_degree_lon:.4f}")
            pixel_width_m = pixel_width * meters_per_degree_lon
            print(f"  - pixel_width_m  = {pixel_width_m:.4f} meter")

            print(f"\n  - pixel_height_m = pixel_height × meters_per_degree_lat")
            print(f"  - pixel_height_m = {pixel_height} × {meters_per_degree_lat}")
            pixel_height_m = pixel_height * meters_per_degree_lat
            print(f"  - pixel_height_m = {pixel_height_m:.4f} meter")

            print(f"\n[STEP 7] Hitung luas 1 pixel:")
            print(f"  - pixel_area_m2 = pixel_width_m × pixel_height_m")
            print(f"  - pixel_area_m2 = {pixel_width_m:.4f} × {pixel_height_m:.4f}")
            pixel_area_m2 = pixel_width_m * pixel_height_m
            print(f"  - pixel_area_m2 = {pixel_area_m2:.6f} m²")

            print(f"\n{'='*80}")
            print(f"HASIL AKHIR:")
            print(f"  Coordinate System: Geographic (WGS 84)")
            print(f"  Resolusi pixel: {pixel_width_m:.4f} m x {pixel_height_m:.4f} m")
            print(f"  Luas per pixel: {pixel_area_m2:.6f} m²")
            print(f"{'='*80}")
        else:
            print(f"\n[STEP 3] Koordinat sudah dalam METER (UTM/Projected)")
            print(f"  - Tidak perlu konversi, langsung kalikan saja!")

            print(f"\n[STEP 4] Hitung luas 1 pixel:")
            print(f"  - pixel_area_m2 = pixel_width × pixel_height")
            print(f"  - pixel_area_m2 = {pixel_width:.4f} × {pixel_height:.4f}")
            pixel_area_m2 = pixel_width * pixel_height
            print(f"  - pixel_area_m2 = {pixel_area_m2:.6f} m²")

            print(f"\n{'='*80}")
            print(f"HASIL AKHIR:")
            print(f"  Resolusi pixel: {pixel_width:.4f} m x {pixel_height:.4f} m")
            print(f"  Luas per pixel: {pixel_area_m2:.6f} m²")
            print(f"{'='*80}")

        # === Statistik per kategori ===
        # Buat mask detail untuk statistik (tetap breakdown seperti sebelumnya untuk statistik)
        mask_red = (ndvi_clipped >= 0) & (ndvi_clipped < 0.21) & valid_mask
        mask_yellow = (ndvi_clipped >= 0.21) & (ndvi_clipped < 0.4) & valid_mask
        mask_light_green = (ndvi_clipped >= 0.4) & (ndvi_clipped < 0.6) & valid_mask
        mask_green = (ndvi_clipped >= 0.6) & (ndvi_clipped < 0.8) & valid_mask
        mask_dark_green = (ndvi_clipped >= 0.8) & (ndvi_clipped <= 1.0) & valid_mask

        print(f"\n=== STATISTIK KLASIFIKASI NDVI ===")

        # Ambil warna dari RdYlGn untuk setiap kategori berdasarkan nilai tengahnya
        categories = [
            ("< 0", "Bukan Vegetasi", "Hitam", mask_black, [0, 0, 0]),
            ("0 - 0.21", "Tidak Sehat", "Merah", mask_red, [int(c*255) for c in cmap(0.105)[:3]]),
            ("0.21 - 0.4", "Kurang Sehat", "Kuning", mask_yellow, [int(c*255) for c in cmap(0.305)[:3]]),
            ("0.4 - 0.6", "Cukup Sehat", "Hijau Muda", mask_light_green, [int(c*255) for c in cmap(0.5)[:3]]),
            ("0.6 - 0.8", "Sehat", "Hijau", mask_green, [int(c*255) for c in cmap(0.7)[:3]]),
            ("0.8 - 1.0", "Sangat Sehat", "Hijau Tua", mask_dark_green, [int(c*255) for c in cmap(0.9)[:3]]),
        ]

        total_valid = np.sum(valid_mask)
        category_data = []

        print(f"\n{'Range NDVI':<15} {'Kategori':<20} {'Warna':<15} {'Pixels':>12} {'Luas (m²)':>15} {'Luas (Ha)':>12} {'%':>8}")
        print(f"{'-'*105}")

        for range_str, kategori, warna, cat_mask, rgb in categories:
            count = np.sum(cat_mask)
            area_m2 = count * pixel_area_m2
            area_ha = area_m2 / 10000
            percentage = (count / total_valid * 100) if total_valid > 0 else 0

            print(f"{range_str:<15} {kategori:<20} {warna:<15} {count:>12,} {area_m2:>15,.2f} {area_ha:>12,.4f} {percentage:>7.2f}%")

            category_data.append({
                'range': range_str,
                'name': kategori,
                'color_name': warna,
                'area_ha': area_ha,
                'percentage': percentage,
                'rgb': [c/255.0 for c in rgb]
            })

        total_area_m2 = total_valid * pixel_area_m2
        print(f"{'-'*105}")
        print(f"{'TOTAL':<36} {total_valid:>12,} {total_area_m2:>15,.2f} {total_area_m2/10000:>12,.4f} {100.0:>7.2f}%")
        print(f"{'Transparent pixels':<36} {np.sum(~valid_mask):>12,}")

        # === Buat visualisasi JPG dengan peta NDVI + colorbar + bar chart ===
        if output_jpg:
            print(f"\nMembuat visualisasi JPG: {output_jpg}")

            fig = plt.figure(figsize=(20, 8))

            # Plot 1: NDVI colored image dengan colorbar
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(rgba)
            ax1.set_title('NDVI Color Map', fontsize=16, fontweight='bold')
            ax1.axis('off')

            # Tambahkan colorbar untuk NDVI
            # Buat custom colormap: Hitam solid untuk < 0, RdYlGn gradient untuk >= 0
            from matplotlib.colors import LinearSegmentedColormap

            # Ambil RdYlGn colormap
            rdylgn = plt.cm.RdYlGn

            # Buat custom colormap dengan 2 segmen:
            # -1 to 0: Hitam solid
            # 0 to 1: RdYlGn gradient
            n_bins = 256
            black_section = int(n_bins * (0 - (-1)) / (1 - (-1)))  # -1 to 0
            gradient_section = n_bins - black_section               # 0 to 1

            colors = []
            # Hitam untuk -1 to 0
            colors.extend([[0, 0, 0, 1]] * black_section)
            # RdYlGn gradient untuk 0 to 1
            for i in range(gradient_section):
                colors.append(rdylgn(i / gradient_section))

            custom_cmap = LinearSegmentedColormap.from_list('custom_ndvi', colors, N=n_bins)

            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', fontsize=11)
            cbar.set_ticks([-1, -0.5, 0, 0.21, 0.4, 0.6, 0.8, 1.0])

            # Tambahkan legend manual dengan kategori
            legend_patches = []
            for d in category_data:
                label = f"{d['range']}: {d['name']}"
                legend_patches.append(mpatches.Patch(color=d['rgb'], label=label))

            ax1.legend(handles=legend_patches, loc='upper left', fontsize=10,
                      framealpha=0.9, title='Klasifikasi NDVI', title_fontsize=11)

            # Plot 2: Bar chart luas per zona
            ax2 = plt.subplot(1, 2, 2)

            # Data untuk bar chart
            names = [d['color_name'] for d in category_data]
            areas_ha = [d['area_ha'] for d in category_data]
            percentages = [d['percentage'] for d in category_data]
            colors_bar = [d['rgb'] for d in category_data]

            # Buat bar chart horizontal
            bars = ax2.barh(names, areas_ha, color=colors_bar, edgecolor='black', linewidth=1)

            # Tambahkan label nilai dan persentase di sebelah bar
            for i, (bar, area, pct) in enumerate(zip(bars, areas_ha, percentages)):
                width = bar.get_width()
                if width > 0:
                    ax2.text(width + max(areas_ha)*0.02, bar.get_y() + bar.get_height()/2,
                            f'{area:.2f} Ha ({pct:.1f}%)',
                            ha='left', va='center', fontsize=10, fontweight='bold')

            ax2.set_xlabel('Area (Hectares)', fontsize=12, fontweight='bold')
            ax2.set_title('Distribusi Luas per Kategori NDVI', fontsize=16, fontweight='bold', pad=20)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)

            # Tambahkan info total luas
            fig.text(0.5, 0.02, f'Total Area: {total_area_m2/10000:.2f} Ha ({total_area_m2:,.0f} m²)',
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            plt.tight_layout(rect=[0, 0.04, 1, 1])
            plt.savefig(output_jpg, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✓ Visualisasi JPG disimpan: {output_jpg}")

        print("\nSelesai!")

if __name__ == "__main__":
    # Contoh penggunaan
    input_file = "/home/adedi/Downloads/ndvi_qgis.tif"
    output_png_file = "/home/adedi/Downloads/ndvi_qgis.png"
    output_tif_file = "/home/adedi/Downloads/ndvi_color.tif"
    output_jpg_file = "/home/adedi/Downloads/ndvi_visualization.jpg"

    apply_custom_ndvi_colormap(input_file, output_png_file, output_tif_file, output_jpg_file)
