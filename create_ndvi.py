import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path ke file orthomosaic
nir_file = "/home/adedi/Documents/Tugas_Akhir/Data/File Agisoft/nir_fix.tif"
red_file = "/home/adedi/Documents/Tugas_Akhir/Data/File Agisoft/red_fix.tif"
output_file = "/home/adedi/Documents/Tugas_Akhir/Computer Vision/NDVI/output/ndvi_orthomosaic.tif"

# Baca file NIR
print("Membaca file NIR...")
with rasterio.open(nir_file) as nir_src:
    print(f"NIR bands: {nir_src.count}")
    nir_band1 = nir_src.read(1).astype(float)
    if nir_src.count > 1:
        nir_band2 = nir_src.read(2).astype(float)
        print(f"NIR Band 1 range: {nir_band1.min()} - {nir_band1.max()}, mean: {nir_band1.mean():.2f}")
        print(f"NIR Band 2 range: {nir_band2.min()} - {nir_band2.max()}, mean: {nir_band2.mean():.2f}")
    nir = nir_band1
    profile = nir_src.profile
    print(f"NIR shape: {nir.shape}")

# Baca file Red
print("Membaca file Red...")
with rasterio.open(red_file) as red_src:
    print(f"Red bands: {red_src.count}")
    red_band1 = red_src.read(1).astype(float)
    if red_src.count > 1:
        red_band2 = red_src.read(2).astype(float)
        print(f"Red Band 1 range: {red_band1.min()} - {red_band1.max()}, mean: {red_band1.mean():.2f}")
        print(f"Red Band 2 range: {red_band2.min()} - {red_band2.max()}, mean: {red_band2.mean():.2f}")
    red = red_band1
    print(f"Red shape: {red.shape}")

# Hitung NDVI
print("Menghitung NDVI...")
# Mask untuk nodata (0 dan 65535)
valid_mask = (nir > 0) & (nir < 65535) & (red > 0) & (red < 65535)

# NDVI = (NIR - Red) / (NIR + Red)
# Tambahkan epsilon untuk hindari division by zero
ndvi = np.zeros_like(nir, dtype=float)
ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / (nir[valid_mask] + red[valid_mask] + 1e-8)

# Clip nilai NDVI ke range -1 sampai 1
ndvi = np.clip(ndvi, -1, 1)

# Set nodata area ke NaN
ndvi[~valid_mask] = np.nan

ndvi_valid = ndvi[~np.isnan(ndvi)]
print(f"NDVI range: {ndvi_valid.min():.4f} - {ndvi_valid.max():.4f}")
print(f"NDVI mean: {ndvi_valid.mean():.4f}")
print(f"Valid pixels: {np.sum(valid_mask)} / {ndvi.size}")

# Simpan hasil NDVI
print(f"Menyimpan NDVI ke {output_file}...")
profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)

print("NDVI orthomosaic berhasil dibuat!")

# Visualisasi
print("Membuat visualisasi...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(nir, cmap='gray')
axes[0].set_title('NIR Band')
axes[0].axis('off')

axes[1].imshow(red, cmap='gray')
axes[1].set_title('Red Band')
axes[1].axis('off')

# Mask NaN values untuk visualisasi
ndvi_plot = np.ma.masked_invalid(ndvi)
im = axes[2].imshow(ndvi_plot, cmap='RdYlGn', vmin=-1, vmax=1)
axes[2].set_title('NDVI')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('/home/adedi/Documents/Tugas_Akhir/Computer Vision/NDVI/output/ndvi_visualization.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan!")
plt.show()
