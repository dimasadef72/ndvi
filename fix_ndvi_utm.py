import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import tempfile

def apply_custom_ndvi_colormap_utm(input_tif, output_png, output_tif, output_jpg=None, target_crs='EPSG:32749'):
    """
    Apply custom colormap to NDVI grayscale .tif (-1 to 1 range) - UTM VERSION
    Output: colored PNG, TIF with transparency preserved, and JPG visualization

    Script ini OTOMATIS convert WGS 84 → UTM jika diperlukan!

    Klasifikasi:
    < 0: Hitam (Bukan Vegetasi)
    >= 0: RdYlGn gradient (Merah → Kuning → Hijau)
      - 0 - 0.21: Merah (Tidak Sehat)
      - 0.21 - 0.4: Kuning (Kurang Sehat)
      - 0.4 - 0.6: Hijau Muda (Cukup Sehat)
      - 0.6 - 0.8: Hijau (Sehat)
      - 0.8 - 1.0: Hijau Tua (Sangat Sehat)

    Parameters:
    - input_tif: path to input NDVI grayscale .tif (bisa WGS 84 atau UTM)
    - output_png: path to output colored PNG
    - output_tif: path to output colored GeoTIFF
    - output_jpg: (optional) path to output visualization JPG with map + colorbar + stats
    - target_crs: (optional) target CRS untuk konversi, default 'EPSG:32749' (UTM Zone 49S - Surabaya)
    """

    print(f"Membaca file: {input_tif}")

    # Baca file NDVI
    with rasterio.open(input_tif) as src_original:
        print(f"\n{'='*80}")
        print(f"CEK SISTEM KOORDINAT")
        print(f"{'='*80}")
        print(f"Input CRS: {src_original.crs}")

        # Cek apakah perlu convert
        need_conversion = src_original.crs and src_original.crs.is_geographic

        if need_conversion:
            print(f"📍 File dalam Geographic (WGS 84) - akan diconvert ke {target_crs}")
            print(f"{'='*80}")

            # Convert ke UTM menggunakan tempfile
            print(f"\n🔄 Converting WGS 84 → {target_crs}...")

            # Hitung transformasi baru
            # convert wgs 84 to utm
            transform, width, height = calculate_default_transform(
                src_original.crs, target_crs,
                src_original.width, src_original.height,
                *src_original.bounds
            )

            # Buat profile baru untuk file UTM
            kwargs = src_original.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Buat temporary file untuk hasil convert
            temp_utm = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_utm_path = temp_utm.name
            temp_utm.close()

            # Reproject ke UTM
            with rasterio.open(temp_utm_path, 'w', **kwargs) as dst:
                for i in range(1, src_original.count + 1):
                    reproject(
                        source=rasterio.band(src_original, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_original.transform,
                        src_crs=src_original.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )

                # Reproject mask juga
                src_mask = src_original.read_masks(1)
                dst_mask = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=src_mask,
                    destination=dst_mask,
                    src_transform=src_original.transform,
                    src_crs=src_original.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                dst.write_mask(dst_mask > 0)

            print(f"✅ Conversion selesai!")
            print(f"   Original: {src_original.width}x{src_original.height} pixels")
            print(f"   UTM:      {width}x{height} pixels")

            # Buka file UTM hasil convert
            src = rasterio.open(temp_utm_path)
            using_temp = True
        else:
            print(f"✅ File sudah dalam sistem proyeksi (UTM/Projected)")
            print(f"{'='*80}")
            src = src_original
            using_temp = False

    # Dari sini, src adalah file dalam UTM (baik original atau hasil convert)
    try:
        ndvi = src.read(1)
        profile = src.profile.copy()

        # Baca mask (area valid vs transparan)
        mask = src.read_masks(1)
        valid_mask = mask > 0

        print(f"\nNDVI range: {np.nanmin(ndvi):.3f} sampai {np.nanmax(ndvi):.3f}")
        print(f"Valid pixels: {np.sum(valid_mask)}, Transparent pixels: {np.sum(~valid_mask)}")

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
        ndvi_for_cmap = np.zeros_like(ndvi_clipped, dtype=float)
        ndvi_for_cmap[mask_vegetation] = ndvi_clipped[mask_vegetation]

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
        img_png = Image.fromarray(rgba)
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
            nodata=None
        )

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            dst.set_band_description(4, 'Alpha')

            dst.write(rgba[:, :, 0], 1)
            dst.write(rgba[:, :, 1], 2)
            dst.write(rgba[:, :, 2], 3)
            dst.write(rgba[:, :, 3], 4)

            dst.write_mask(rgba[:, :, 3] > 0)

        print(f"✓ GeoTIFF disimpan: {output_tif}")

        # === Hitung luas per pixel (SIMPLE untuk UTM!) ===
        print(f"\n{'='*80}")
        print(f"PERHITUNGAN LUAS PER PIXEL (UTM - SIMPLE METHOD)")
        print(f"{'='*80}")

        pixel_width = abs(src.transform[0])
        pixel_height = abs(src.transform[4])

        print(f"\n✅ Koordinat sudah dalam METER (UTM/Projected)")
        print(f"   Tidak perlu konversi degree → meter!")
        print(f"\n📏 Ukuran pixel:")
        print(f"   - pixel_width  = {pixel_width:.4f} meter")
        print(f"   - pixel_height = {pixel_height:.4f} meter")

        # Langsung kalikan!
        pixel_area_m2 = pixel_width * pixel_height

        print(f"\n📐 Luas per pixel:")
        print(f"   - pixel_area_m2 = pixel_width × pixel_height")
        print(f"   - pixel_area_m2 = {pixel_width:.4f} × {pixel_height:.4f}")
        print(f"   - pixel_area_m2 = {pixel_area_m2:.6f} m²")
        print(f"\n{'='*80}")

        # === Statistik per kategori ===
        # Buat mask detail untuk statistik
        mask_red = (ndvi_clipped >= 0) & (ndvi_clipped < 0.21) & valid_mask
        mask_yellow = (ndvi_clipped >= 0.21) & (ndvi_clipped < 0.4) & valid_mask
        mask_light_green = (ndvi_clipped >= 0.4) & (ndvi_clipped < 0.6) & valid_mask
        mask_green = (ndvi_clipped >= 0.6) & (ndvi_clipped < 0.8) & valid_mask
        mask_dark_green = (ndvi_clipped >= 0.8) & (ndvi_clipped <= 1.0) & valid_mask

        print(f"\n=== STATISTIK KLASIFIKASI NDVI ===")

        # Ambil warna dari RdYlGn untuk setiap kategori
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

            # SIMPLE: Langsung kalikan!
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

        # === Buat visualisasi JPG ===
        if output_jpg:
            print(f"\nMembuat visualisasi JPG: {output_jpg}")

            fig = plt.figure(figsize=(20, 8))

            # Plot 1: NDVI colored image dengan colorbar
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(rgba)
            ax1.set_title('NDVI Color Map (UTM)', fontsize=16, fontweight='bold')
            ax1.axis('off')

            # Tambahkan colorbar
            from matplotlib.colors import LinearSegmentedColormap

            rdylgn = plt.cm.RdYlGn
            n_bins = 256
            black_section = int(n_bins * 0.5)
            gradient_section = n_bins - black_section

            colors = []
            colors.extend([[0, 0, 0, 1]] * black_section)
            for i in range(gradient_section):
                colors.append(rdylgn(i / gradient_section))

            custom_cmap = LinearSegmentedColormap.from_list('custom_ndvi', colors, N=n_bins)

            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI Value', fontsize=11)
            cbar.set_ticks([-1, -0.5, 0, 0.21, 0.4, 0.6, 0.8, 1.0])

            # Tambahkan legend
            legend_patches = []
            for d in category_data:
                label = f"{d['range']}: {d['name']}"
                legend_patches.append(mpatches.Patch(color=d['rgb'], label=label))

            ax1.legend(handles=legend_patches, loc='upper left', fontsize=10,
                      framealpha=0.9, title='Klasifikasi NDVI', title_fontsize=11)

            # Plot 2: Bar chart
            ax2 = plt.subplot(1, 2, 2)

            names = [d['color_name'] for d in category_data]
            areas_ha = [d['area_ha'] for d in category_data]
            percentages = [d['percentage'] for d in category_data]
            colors_bar = [d['rgb'] for d in category_data]

            bars = ax2.barh(names, areas_ha, color=colors_bar, edgecolor='black', linewidth=1)

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

            fig.text(0.5, 0.02, f'Total Area: {total_area_m2/10000:.2f} Ha ({total_area_m2:,.0f} m²) | UTM Projection',
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            plt.tight_layout(rect=[0, 0.04, 1, 1])
            plt.savefig(output_jpg, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✓ Visualisasi JPG disimpan: {output_jpg}")

        print("\n✅ Selesai!")
        print("\n💡 Keuntungan UTM:")
        print("   - Perhitungan luas: SIMPLE (langsung kalikan pixel_width × pixel_height)")
        print("   - Tidak perlu konversi degree → meter")
        print("   - Tidak perlu cos(latitude)")
        print("   - Lebih akurat untuk area lokal")

    finally:
        # Cleanup: tutup file dan hapus temp file jika ada
        if 'src' in locals() and 'src_original' in locals() and src != src_original:
            src.close()

        if 'using_temp' in locals() and using_temp and 'temp_utm_path' in locals():
            try:
                os.unlink(temp_utm_path)
                print(f"\n🗑️  Temp file cleaned up")
            except:
                pass

if __name__ == "__main__":
    # Contoh penggunaan
    print("="*80)
    print("NDVI COLORMAP - UTM VERSION (AUTO CONVERT)")
    print("="*80)
    print("\n✨ Script ini OTOMATIS convert WGS 84 → UTM!")
    print("   Input bisa WGS 84 (EPSG:4326) atau UTM (EPSG:32749)")
    print("   Perhitungan luas akan SELALU dalam UTM (simple & akurat)")
    print("\n" + "="*80 + "\n")

    # File input bisa WGS 84 atau UTM, script akan handle otomatis!
    input_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/sawah3_clipped.tif"
    output_png_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Clipped/sawah3_clipped.png"
    output_tif_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Clipped/sawah3_clipped.tif"
    output_jpg_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Clipped/visual_sawah3_clipped.jpg"

    apply_custom_ndvi_colormap_utm(input_file, output_png_file, output_tif_file, output_jpg_file)
