import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np

def remap_ndvi_for_colormap(ndvi_values):
    """Custom remapping dari fix_ndvi_custom.py"""
    remapped = np.zeros_like(ndvi_values, dtype=float)

    # NDVI 0.0 - 0.6 -> RdYlGn 0.0 - 0.2
    mask1 = (ndvi_values >= 0) & (ndvi_values < 0.6)
    remapped[mask1] = ndvi_values[mask1] / 0.6 * 0.2

    # NDVI 0.6 - 0.8 -> RdYlGn 0.2 - 0.5
    mask2 = (ndvi_values >= 0.6) & (ndvi_values < 0.8)
    remapped[mask2] = 0.2 + (ndvi_values[mask2] - 0.6) / 0.2 * 0.3

    # NDVI 0.8 - 1.0 -> RdYlGn 0.5 - 1.0
    mask3 = (ndvi_values >= 0.8) & (ndvi_values <= 1.0)
    remapped[mask3] = 0.5 + (ndvi_values[mask3] - 0.8) / 0.2 * 0.5

    return remapped

# Get RdYlGn colormap
cmap = cm.get_cmap('RdYlGn')

# NDVI values to check (asli)
ndvi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Remap ke custom values
ndvi_array = np.array(ndvi_values)
remapped_values = remap_ndvi_for_colormap(ndvi_array)

print("Custom NDVI Remapping - Color Values")
print("="*100)
print(f"{'NDVI':<10} {'RdYlGn':<12} {'RGB (0-1)':<35} {'RGB (0-255)':<20} {'Hex':<10} {'Warna'}")
print("="*100)

# Nama warna untuk setiap value
color_names = {
    0.0: "Merah Marun",
    0.1: "Merah Cerah",
    0.2: "Oranye Kemerahan",
    0.3: "Oranye Terang",
    0.4: "Kuning Oranye",
    0.5: "Kuning Muda",
    0.6: "Kuning Kehijauan",
    0.7: "Hijau Muda",
    0.8: "Hijau Sedang",
    0.9: "Hijau Tua",
    1.0: "Hijau Gelap"
}

for ndvi_val, remapped_val in zip(ndvi_values, remapped_values):
    color = cmap(remapped_val)
    rgb_0_1 = color[:3]
    rgb_0_255 = tuple(int(c * 255) for c in rgb_0_1)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_0_255)

    # Cari nama warna berdasarkan remapped value yang terdekat
    closest_key = min(color_names.keys(), key=lambda x: abs(remapped_val - x))
    warna = color_names[closest_key]

    print(f"{ndvi_val:<10.1f} {remapped_val:<12.2f} {str(rgb_0_1):<35} {str(rgb_0_255):<20} {hex_color:<10} {warna}")

print("="*100)
print("\nKeterangan:")
print("  SEGMEN 1: NDVI 0.0 - 0.6  → RdYlGn 0.00 - 0.20 (Merah Tua → Merah)")
print("  SEGMEN 2: NDVI 0.6 - 0.8  → RdYlGn 0.20 - 0.50 (Merah → Oranye → Kuning)")
print("  SEGMEN 3: NDVI 0.8 - 1.0  → RdYlGn 0.50 - 1.00 (Kuning → Hijau Tua)")

# Create visualization
fig, ax = plt.subplots(figsize=(14, 10))

# Draw color boxes
for i, (ndvi_val, remapped_val) in enumerate(zip(ndvi_values, remapped_values)):
    color = cmap(remapped_val)
    rgb_0_1 = color[:3]
    rgb_0_255 = tuple(int(c * 255) for c in rgb_0_1)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_0_255)

    # Draw colored rectangle
    rect = patches.Rectangle((0, 10-i), 2.5, 0.8,
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1.5)
    ax.add_patch(rect)

    # Add NDVI label (kiri)
    ax.text(-0.5, 10-i+0.4, f'NDVI {ndvi_val:.1f}',
            va='center', ha='right', fontsize=11, weight='bold')

    # Add remapped value and color info (kanan)
    ax.text(3.0, 10-i+0.4,
            f'→ RdYlGn {remapped_val:.2f}  |  RGB(0-255): {rgb_0_255}  |  Hex: {hex_color}',
            va='center', ha='left', fontsize=9, family='monospace')

# Add separator lines
ax.axhline(y=5, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(y=3, color='orange', linestyle='--', linewidth=2, alpha=0.5)

# Add annotations
ax.text(-0.5, 8, 'SEGMEN 1', ha='right', fontsize=10, style='italic', color='red', weight='bold')
ax.text(-0.5, 4, 'SEGMEN 2', ha='right', fontsize=10, style='italic', color='orange', weight='bold')
ax.text(-0.5, 1.5, 'SEGMEN 3', ha='right', fontsize=10, style='italic', color='green', weight='bold')

# Set axis properties
ax.set_xlim(-2, 14)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Custom NDVI Remapping - Visual Color Reference\n(NDVI → Custom RdYlGn Value → Actual Color)',
             fontsize=14, weight='bold', pad=20)

# Add legend
legend_text = (
    "SEGMEN 1 (Merah): NDVI 0.0-0.6 → compressed ke RdYlGn 0.0-0.2\n"
    "SEGMEN 2 (Kuning): NDVI 0.6-0.8 → mapped ke RdYlGn 0.2-0.5\n"
    "SEGMEN 3 (Hijau): NDVI 0.8-1.0 → expanded ke RdYlGn 0.5-1.0"
)
ax.text(7, 0.5, legend_text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('custom_ndvi_color_values.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: custom_ndvi_color_values.png")
plt.show()
