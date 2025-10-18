import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

def apply_colormap_to_ndvi(input_tif, output_tif, output_png=None):
    """
    Mengaplikasikan color ramp ke NDVI grayscale

    Parameters:
    - input_tif: Path ke file NDVI grayscale (.tif)
    - output_tif: Path output file berwarna (.tif)
    - output_png: (Optional) Path output preview PNG
    """
    print(f"Membaca file: {input_tif}")

    # Baca NDVI grayscale
    with rasterio.open(input_tif) as src:
        ndvi = src.read(1)
        profile = src.profile.copy()
        nodata_value = src.nodata

        print(f"NDVI range: {np.nanmin(ndvi):.3f} sampai {np.nanmax(ndvi):.3f}")
        print(f"Nodata value: {nodata_value}")
        print(f"Band count: {src.count}")

        # Baca internal mask dari rasterio (ini yang penting!)
        # Mask: 255 = valid, 0 = nodata/transparan
        mask = src.read_masks(1)
        valid_mask = mask > 0
        print(f"Valid pixels: {np.sum(valid_mask)}, Transparent pixels: {np.sum(~valid_mask)}")

        # Clip NDVI ke range [0, 1]
        ndvi_clipped = np.clip(ndvi, 0, 1)

        # Gunakan colormap RdYlGn
        cmap = plt.cm.RdYlGn

        # Aplikasikan colormap (hasilnya RGBA)
        ndvi_colored = cmap(ndvi_clipped)

        # Konversi ke RGBA (0-255)
        rgba = (ndvi_colored * 255).astype(np.uint8)

        # Set alpha channel: 255 untuk valid, 0 untuk transparan
        rgba[:, :, 3] = np.where(valid_mask, 255, 0)

        # Set RGB ke putih untuk area transparan (agar tidak hitam)
        rgba[:, :, 0] = np.where(valid_mask, rgba[:, :, 0], 255)
        rgba[:, :, 1] = np.where(valid_mask, rgba[:, :, 1], 255)
        rgba[:, :, 2] = np.where(valid_mask, rgba[:, :, 2], 255)

        # AUTO-CROP: Hanya ambil area yang valid (menghilangkan area transparan)
        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Crop RGBA dan mask
        rgba_cropped = rgba[row_min:row_max+1, col_min:col_max+1]
        valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]

        # PENTING: Set alpha channel ke 255 untuk SEMUA pixel (hilangkan transparansi)
        rgba_cropped[:, :, 3] = 255

        # Update transform untuk cropped extent
        from rasterio.transform import from_bounds
        transform = src.transform
        new_transform = rasterio.windows.transform(
            rasterio.windows.Window(col_min, row_min, col_max-col_min+1, row_max-row_min+1),
            transform
        )

        print(f"Original size: {rgba.shape[:2]}")
        print(f"Cropped size: {rgba_cropped.shape[:2]}")
        print(f"Crop bounds: rows [{row_min}:{row_max+1}], cols [{col_min}:{col_max+1}]")

        # Update profile untuk RGBA cropped
        profile.update(
            dtype=rasterio.uint8,
            count=4,
            photometric='RGB',
            nodata=None,  # Alpha channel handles transparency
            width=rgba_cropped.shape[1],
            height=rgba_cropped.shape[0],
            transform=new_transform
        )

        # Simpan sebagai GeoTIFF berwarna dengan transparansi (CROPPED)
        print(f"Menyimpan GeoTIFF berwarna: {output_tif}")
        with rasterio.open(output_tif, 'w', **profile) as dst:
            # Set nodata untuk setiap band agar GIS software bisa recognize
            for i in range(1, 5):
                dst.set_band_description(i, ['Red', 'Green', 'Blue', 'Alpha'][i-1])

            dst.write(rgba_cropped[:, :, 0], 1)  # Red
            dst.write(rgba_cropped[:, :, 1], 2)  # Green
            dst.write(rgba_cropped[:, :, 2], 3)  # Blue
            dst.write(rgba_cropped[:, :, 3], 4)  # Alpha

        # Simpan preview PNG jika diminta
        if output_png:
            print(f"Menyimpan preview PNG: {output_png}")

            # Konversi RGBA ke RGB (buang alpha channel untuk PNG)
            # Ambil hanya RGB channels
            rgb_cropped = rgba_cropped[:, :, :3]

            # Simpan PNG langsung dengan PIL sebagai RGB (tanpa transparansi)
            img = Image.fromarray(rgb_cropped, mode='RGB')
            img.save(output_png)

            # Tambahkan versi dengan colorbar menggunakan matplotlib
            output_with_colorbar = output_png.replace('.png', '_with_colorbar.png')
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.patch.set_facecolor('white')  # Background putih
            ax.imshow(rgb_cropped)  # Gunakan RGB, bukan RGBA
            ax.axis('off')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', fontsize=12)

            plt.tight_layout(pad=0)
            plt.savefig(output_with_colorbar, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Colorbar version saved: {output_with_colorbar}")

        # HITUNG LUAS PER KATEGORI NDVI
        print(f"\n{'='*70}")
        print(f"ANALISIS LUAS NDVI")
        print(f"{'='*70}")

        # Hitung pixel resolution (luas per pixel dalam m²)
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])

        # Cek apakah dalam degrees atau meter
        if src.crs and src.crs.is_geographic:
            # Koordinat geografis (degrees) - konversi ke meter
            # Gunakan aproksimasi: 1 degree ≈ 111,320 meter di equator
            # Untuk luas yang lebih akurat, gunakan latitude tengah
            import math
            bounds = src.bounds
            lat_center = (bounds.top + bounds.bottom) / 2

            # Konversi degree ke meter
            meters_per_degree_lon = 111320 * math.cos(math.radians(lat_center))
            meters_per_degree_lat = 111320

            pixel_width_m = pixel_width * meters_per_degree_lon
            pixel_height_m = pixel_height * meters_per_degree_lat
            pixel_area_m2 = pixel_width_m * pixel_height_m

            print(f"Coordinate System: Geographic (WGS 84)")
            print(f"Latitude tengah: {lat_center:.6f}°")
            print(f"Resolusi pixel: {pixel_width:.10f}° x {pixel_height:.10f}°")
            print(f"Resolusi pixel (meter): {pixel_width_m:.4f} m x {pixel_height_m:.4f} m")
            print(f"Luas per pixel: {pixel_area_m2:.6f} m²")
        else:
            # Koordinat proyeksi (sudah dalam meter)
            pixel_area_m2 = pixel_width * pixel_height
            print(f"Resolusi pixel: {pixel_width:.4f} m x {pixel_height:.4f} m")
            print(f"Luas per pixel: {pixel_area_m2:.6f} m²")

        # Filter hanya pixel valid
        ndvi_valid = ndvi[valid_mask]

        # Definisi kategori NDVI
        categories = [
            ("Merah - Sangat Rendah", -1.0, 0.0),
            ("Oranye - Rendah", 0.0, 0.2),
            ("Kuning - Sedang Rendah", 0.2, 0.4),
            ("Hijau Muda - Sedang", 0.4, 0.6),
            ("Hijau - Baik", 0.6, 0.8),
            ("Hijau Tua - Sangat Baik", 0.8, 1.0),
        ]

        print(f"\n{'Kategori':<30} {'Jumlah Pixel':>15} {'Luas (m²)':>15} {'Luas (Ha)':>10} {'Persentase':>12}")
        print(f"{'-'*85}")

        total_pixels = len(ndvi_valid)
        for name, min_val, max_val in categories:
            if max_val == 1.0:
                count = np.sum((ndvi_valid >= min_val) & (ndvi_valid <= max_val))
            else:
                count = np.sum((ndvi_valid >= min_val) & (ndvi_valid < max_val))

            area_m2 = count * pixel_area_m2
            area_ha = area_m2 / 10000
            percentage = (count / total_pixels) * 100

            print(f"{name:<30} {count:>15,} {area_m2:>15,.2f} {area_ha:>10,.4f} {percentage:>11.2f}%")

        total_area_m2 = total_pixels * pixel_area_m2
        print(f"{'-'*85}")
        print(f"{'TOTAL':<30} {total_pixels:>15,} {total_area_m2:>15,.2f} {total_area_m2/10000:>10,.4f} {100.0:>11.2f}%")
        print(f"{'='*70}")

        # VISUALISASI LUAS DENGAN CHART
        if output_png:
            # Simpan data untuk visualisasi
            category_data = []
            category_colors = []

            for name, min_val, max_val in categories:
                if max_val == 1.0:
                    count = np.sum((ndvi_valid >= min_val) & (ndvi_valid <= max_val))
                else:
                    count = np.sum((ndvi_valid >= min_val) & (ndvi_valid < max_val))

                area_ha = (count * pixel_area_m2) / 10000
                percentage = (count / total_pixels) * 100

                category_data.append({
                    'name': name,
                    'area_ha': area_ha,
                    'percentage': percentage,
                    'mid_val': (min_val + max_val) / 2
                })

            # Buat visualisasi dengan 2 panel: peta NDVI + pie chart
            output_with_stats = output_png.replace('.png', '_with_stats.png')
            fig = plt.figure(figsize=(18, 8))

            # Panel 1: Peta NDVI dengan colorbar
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(rgb_cropped)
            ax1.axis('off')
            ax1.set_title('NDVI Map', fontsize=14, fontweight='bold')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', fontsize=11)

            # Panel 2: Bar chart dengan statistik luas
            ax2 = plt.subplot(1, 2, 2)

            # Data untuk bar chart
            names = [d['name'].split(' - ')[0] for d in category_data]  # Ambil hanya warna
            areas_ha = [d['area_ha'] for d in category_data]
            percentages = [d['percentage'] for d in category_data]
            colors_bar = [cmap(d['mid_val']) for d in category_data]

            # Buat bar chart
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
            plt.savefig(output_with_stats, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Statistics visualization saved: {output_with_stats}")

    print("\nSelesai!")

if __name__ == "__main__":
    # Contoh penggunaan
    input_file = "/home/adedi/Downloads/NDVI.tif"
    output_file = "/home/adedi/Downloads/NDVI_color.tif"
    preview_file = "/home/adedi/Downloads/ndvi_preview.png"

    apply_colormap_to_ndvi(input_file, output_file, preview_file)
