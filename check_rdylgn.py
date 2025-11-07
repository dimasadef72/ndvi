import matplotlib.pyplot as plt
import numpy as np

cmap = plt.cm.RdYlGn

print('RdYlGn colormap default gradient:')
print('='*70)
print(f'{"Value":<10} {"RGB (0-1)":<30} {"RGB (0-255)":<20} {"Hex"}')
print('='*70)

for val in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rgb = cmap(val)[:3]
    rgb8 = tuple(int(c * 255) for c in rgb)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb8)
    print(f'{val:<10.1f} ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})  {str(rgb8):<20} {hex_color}')

print('='*70)
print('\nKeterangan:')
print('  0.0 - 0.2  : Merah (Red)')
print('  0.2 - 0.5  : Kuning (Yellow)')
print('  0.5 - 1.0  : Hijau (Green)')
