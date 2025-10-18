import rasterio
import numpy as np
import matplotlib.pyplot as plt

def ndvi_to_colored_tif(input_tif, output_tif):
    """
    Convert NDVI grayscale .tif to colored .tif with color ramp

    Parameters:
    - input_tif: path to input NDVI .tif file
    - output_tif: path to output colored .tif file
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

        print("\nNDVI values di area 10x10 (row 0-9, col 0-9):")
        print(ndvi[0:10, 0:10])
        print("\nMask values di area 10x10 (0=transparan, 255=valid):")
        print(mask[0:10, 0:10])

        # Handle NaN dan invalid values
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip NDVI ke range [0, 1] untuk colormap
        ndvi_clipped = np.clip(ndvi_clean, 0, 1)

        # Gunakan colormap RdYlGn (Red-Yellow-Green)
        cmap = plt.cm.RdYlGn

        # Aplikasikan colormap (hasilnya RGBA float 0-1)
        ndvi_colored = cmap(ndvi_clipped)

        # Konversi ke RGBA uint8 (0-255)
        rgba = (ndvi_colored * 255).astype(np.uint8)

        # Set alpha channel dan RGB ke 0 untuk area transparan
        # - Transparan (alpha=0, RGB=0) jika: area di luar pemetaan
        # - Opaque (alpha=255) jika: area dalam pemetaan - kasih warna sesuai NDVI
        rgba[:, :, 3] = np.where(valid_mask, 255, 0)

        # Set RGB ke 0 untuk area transparan supaya tidak ada warna merah
        for i in range(3):
            rgba[:, :, i] = np.where(valid_mask, rgba[:, :, i], 0)

        # Update profile untuk RGBA
        profile.update(
            dtype=rasterio.uint8,
            count=4,
            photometric='RGB',
            compress='lzw'
        )

        # Simpan sebagai GeoTIFF berwarna dengan transparansi
        print(f"Menyimpan GeoTIFF berwarna: {output_tif}")
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(rgba[:, :, 0], 1)  # Red
            dst.write(rgba[:, :, 1], 2)  # Green
            dst.write(rgba[:, :, 2], 3)  # Blue
            dst.write(rgba[:, :, 3], 4)  # Alpha

            # Set mask untuk band alpha - ini yang bikin QGIS deteksi transparansi
            dst.write_mask(rgba[:, :, 3] > 0)

        print(f"Berhasil convert {input_tif} -> {output_tif}")

        # Hitung pixel resolution (luas per pixel dalam m²)
        pixel_width = abs(src.transform[0])
        pixel_height = abs(src.transform[4])

        # Cek apakah dalam degrees atau meter
        if src.crs and src.crs.is_geographic:
            # Koordinat geografis (degrees) - konversi ke meter
            import math
            bounds = src.bounds
            lat_center = (bounds.top + bounds.bottom) / 2

            # Konversi degree ke meter
            meters_per_degree_lon = 111320 * math.cos(math.radians(lat_center))
            meters_per_degree_lat = 111320

            pixel_width_m = pixel_width * meters_per_degree_lon
            pixel_height_m = pixel_height * meters_per_degree_lat
            pixel_area_m2 = pixel_width_m * pixel_height_m

            print(f"\nCoordinate System: Geographic (WGS 84)")
            print(f"Latitude tengah: {lat_center:.6f}°")
            print(f"Resolusi pixel: {pixel_width:.10f}° x {pixel_height:.10f}°")
            print(f"Resolusi pixel (meter): {pixel_width_m:.4f} m x {pixel_height_m:.4f} m")
            print(f"Luas per pixel: {pixel_area_m2:.6f} m²")
        else:
            # Koordinat proyeksi (sudah dalam meter)
            pixel_area_m2 = pixel_width * pixel_height
            print(f"\nResolusi pixel: {pixel_width:.4f} m x {pixel_height:.4f} m")
            print(f"Luas per pixel: {pixel_area_m2:.6f} m²")

        # Filter hanya pixel valid
        ndvi_valid = ndvi_clean[valid_mask]

        # Definisi kategori NDVI (6 kategori seperti di apply_ndvi_colormap.py)
        categories = [
            ("Merah - Sangat Rendah", -1.0, 0.0),
            ("Oranye - Rendah", 0.0, 0.2),
            ("Kuning - Sedang Rendah", 0.2, 0.4),
            ("Hijau Muda - Sedang", 0.4, 0.6),
            ("Hijau - Baik", 0.6, 0.8),
            ("Hijau Tua - Sangat Baik", 0.8, 1.0),
        ]

        print(f"\n=== PERHITUNGAN LUAS PER ZONA NDVI ===")
        print(f"\n{'Kategori':<30} {'Jumlah Pixel':>15} {'Luas (m²)':>15} {'Luas (Ha)':>10} {'Persentase':>12}")
        print(f"{'-'*85}")

        total_pixels = len(ndvi_valid)
        category_data = []

        for name, min_val, max_val in categories:
            if max_val == 1.0:
                count = np.sum((ndvi_valid >= min_val) & (ndvi_valid <= max_val))
            else:
                count = np.sum((ndvi_valid >= min_val) & (ndvi_valid < max_val))

            area_m2 = count * pixel_area_m2
            area_ha = area_m2 / 10000
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0

            print(f"{name:<30} {count:>15,} {area_m2:>15,.2f} {area_ha:>10,.4f} {percentage:>11.2f}%")

            category_data.append({
                'name': name,
                'area_ha': area_ha,
                'percentage': percentage,
                'mid_val': (min_val + max_val) / 2
            })

        total_area_m2 = total_pixels * pixel_area_m2
        print(f"{'-'*85}")
        print(f"{'TOTAL':<30} {total_pixels:>15,} {total_area_m2:>15,.2f} {total_area_m2/10000:>10,.4f} {100.0:>11.2f}%")

        # Buat visualisasi PNG dengan gambar NDVI + bar chart
        png_output = output_tif.replace('.tif', '_visualization.png')

        fig = plt.figure(figsize=(18, 8))

        # Plot 1: NDVI colored image dengan colorbar
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(rgba)
        ax1.set_title('NDVI Color Map', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Tambahkan colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI Value', fontsize=11)

        # Plot 2: Bar chart luas per zona
        ax2 = plt.subplot(1, 2, 2)

        # Data untuk bar chart
        names = [d['name'].split(' - ')[0] for d in category_data]  # Ambil hanya warna
        areas_ha = [d['area_ha'] for d in category_data]
        percentages = [d['percentage'] for d in category_data]
        colors_bar = [cmap(d['mid_val']) for d in category_data]

        # Buat bar chart horizontal
        bars = ax2.barh(names, areas_ha, color=colors_bar, edgecolor='black', linewidth=0.5)

        # Tambahkan label nilai dan persentase di dalam bar
        for i, (bar, area, pct) in enumerate(zip(bars, areas_ha, percentages)):
            width = bar.get_width()
            if width > 0:
                ax2.text(width/2, bar.get_y() + bar.get_height()/2,
                        f'{area:.2f} Ha\n({pct:.1f}%)',
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white', bbox=dict(boxstyle='round,pad=0.3',
                                                 facecolor='black', alpha=0.6))

        ax2.set_xlabel('Area (Hectares)', fontsize=11, fontweight='bold')
        ax2.set_title('Area Distribution by NDVI Category', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)

        # Tambahkan info total luas
        fig.text(0.5, 0.02, f'Total Area: {total_area_m2/10000:.2f} Ha ({total_area_m2:,.0f} m²)',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(png_output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\nVisualisasi disimpan: {png_output}")

# Contoh penggunaan
if __name__ == "__main__":
    input_file = "/home/adedi/Downloads/ndvi_qgis.tif"
    output_file = "/home/adedi/Downloads/NDVI_vers.tif"

    ndvi_to_colored_tif(input_file, output_file)
