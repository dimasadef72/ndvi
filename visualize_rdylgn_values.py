import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

# Get RdYlGn colormap
cmap = cm.get_cmap('RdYlGn')

# Values to check
values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Draw color boxes for each value
for i, val in enumerate(values):
    # Get color from colormap
    color = cmap(val)
    rgb_0_1 = color[:3]
    rgb_0_255 = tuple(int(c * 255) for c in rgb_0_1)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_0_255)

    # Draw colored rectangle
    rect = patches.Rectangle((0, 10-i), 2, 0.8,
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1)
    ax.add_patch(rect)

    # Add text labels
    ax.text(-0.5, 10-i+0.4, f'{val:.1f}',
            va='center', ha='right', fontsize=11, weight='bold')
    ax.text(2.5, 10-i+0.4,
            f'RGB(0-1): {rgb_0_1}\nRGB(0-255): {rgb_0_255}\nHex: {hex_color}',
            va='center', ha='left', fontsize=9, family='monospace')

# Set axis properties
ax.set_xlim(-1, 10)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('RdYlGn Colormap - Visual Color Reference',
             fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('rdylgn_color_values.png', dpi=150, bbox_inches='tight')
print("✓ Saved: rdylgn_color_values.png")
plt.show()
