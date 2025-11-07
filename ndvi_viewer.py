import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Baca file GeoTIFF NDVI
ndvi_file = "/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/sawah2.tif"

with rasterio.open(ndvi_file) as src:
    ndvi = src.read(1)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(ndvi, cmap='gray')
ax.set_title('NDVI Viewer', fontsize=14, fontweight='bold')
ax.set_xlabel('X (Kolom)')
ax.set_ylabel('Y (Baris)')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Nilai NDVI', rotation=270, labelpad=20)

# Text untuk menampilkan informasi
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='yellow', alpha=0.8), fontsize=11, fontweight='bold')

def on_mouse_move(event):
    """Handler untuk event mouse move"""
    if event.inaxes == ax:
        # Dapatkan posisi piksel
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)

        # Pastikan posisi dalam batas array
        if 0 <= y < ndvi.shape[0] and 0 <= x < ndvi.shape[1]:
            ndvi_value = ndvi[y, x]

            # Update text informasi
            info_text.set_text(
                f'X: {x}, Y: {y}\n'
                f'NDVI: {ndvi_value:.6f}'
            )
        else:
            info_text.set_text('')

        fig.canvas.draw_idle()

# Connect event handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.tight_layout()
plt.show()
