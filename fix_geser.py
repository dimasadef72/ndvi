import rasterio
from rasterio.transform import Affine
import sys

def fix_gps_shift(input_tif, output_tif, shift_north_meter=4.0, shift_east_meter=1.0):
    """
    Fix GPS shift dengan menggeser koordinat

    Parameters:
    - input_tif: file yang perlu dishift
    - output_tif: output file
    - shift_north_meter: jarak shift ke utara (meter), positif = utara, negatif = selatan
    - shift_east_meter: jarak shift ke timur (meter), positif = timur, negatif = barat
    """

    print(f"\n{'='*80}")
    print(f"FIX GPS SHIFT")
    print(f"{'='*80}\n")
    print(f"Input:  {input_tif}")
    print(f"Output: {output_tif}")
    print(f"Shift:  {shift_north_meter} meter ke utara, {shift_east_meter} meter ke timur\n")

    with rasterio.open(input_tif) as src:
        print(f"CRS: {src.crs}")
        print(f"Bounds lama: {src.bounds}")

        # Hitung shift dalam degrees
        # 1 degree latitude ≈ 111,320 meter
        # 1 degree longitude ≈ 111,320 * cos(latitude) meter
        import math
        bounds = src.bounds
        lat_center = (bounds.top + bounds.bottom) / 2
        lat_center_rad = math.radians(lat_center)
        cos_lat = math.cos(lat_center_rad)

        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * cos_lat

        shift_lat_degrees = shift_north_meter / meters_per_degree_lat
        shift_lon_degrees = shift_east_meter / meters_per_degree_lon

        print(f"\nKonversi shift:")
        print(f"  Latitude center: {lat_center:.6f}°")
        print(f"  Utara {shift_north_meter}m = {shift_lat_degrees:.8f}°")
        print(f"  Timur {shift_east_meter}m = {shift_lon_degrees:.8f}°\n")

        # Get transform lama
        old_transform = src.transform

        # Buat transform baru dengan offset latitude DAN longitude
        new_transform = Affine(
            old_transform.a,  # pixel width
            old_transform.b,  # rotation
            old_transform.c + shift_lon_degrees,  # x origin (longitude) - SHIFT KE TIMUR
            old_transform.d,  # rotation
            old_transform.e,  # pixel height
            old_transform.f + shift_lat_degrees  # y origin (latitude) - SHIFT KE UTARA
        )

        # Update profile
        profile = src.profile.copy()
        profile['transform'] = new_transform

        # Copy data ke file baru
        print(f"Menyimpan file baru...")
        with rasterio.open(output_tif, 'w', **profile) as dst:
            # Copy semua band
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)

            # Copy mask jika ada
            if src.read_masks(1) is not None:
                dst.write_mask(src.read_masks(1))

        print(f"✓ File tersimpan: {output_tif}")

        # Cek bounds baru
        with rasterio.open(output_tif) as check:
            print(f"\nBounds baru: {check.bounds}")
            print(f"Selisih latitude: {check.bounds.top - src.bounds.top:.8f} degrees")
            print(f"                  ≈ {(check.bounds.top - src.bounds.top) * 111320:.2f} meter\n")

        print(f"{'='*80}")
        print("✓ SELESAI")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # Default files
    default_input = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/rute_its.tif"
    default_output = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/rute_its_fixed.tif"
    default_shift_north = 5.0  # meter ke utara
    default_shift_east = 1.0   # meter ke timur

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.tif', '_fixed.tif')
        shift_north = float(sys.argv[3]) if len(sys.argv) > 3 else default_shift_north
        shift_east = float(sys.argv[4]) if len(sys.argv) > 4 else default_shift_east
    else:
        input_file = default_input
        output_file = default_output
        shift_north = default_shift_north
        shift_east = default_shift_east
        print(f"ℹ️  Menggunakan file default")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Shift:  {shift_north}m utara, {shift_east}m timur")
        print(f"\n   Atau jalankan: python fix_geser.py <input.tif> [output.tif] [shift_north] [shift_east]\n")

    fix_gps_shift(input_file, output_file, shift_north, shift_east)
