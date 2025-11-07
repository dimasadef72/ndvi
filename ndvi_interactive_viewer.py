import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class NDVIViewer:
    def __init__(self, tif_path):
        self.tif_path = tif_path
        self.src = rasterio.open(tif_path)
        self.ndvi_data = self.src.read(1)
        self.height, self.width = self.ndvi_data.shape

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.im = self.ax.imshow(self.ndvi_data, cmap='RdYlGn', vmin=-1, vmax=1)
        self.ax.set_title(f'NDVI Viewer - Klik untuk melihat info pixel\n{tif_path}')
        self.ax.set_xlabel('Column (X)')
        self.ax.set_ylabel('Row (Y)')

        # Add colorbar
        cbar = plt.colorbar(self.im, ax=self.ax)
        cbar.set_label('NDVI Value')

        # Text untuk info
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', bbox=dict(boxstyle='round',
                                      facecolor='wheat', alpha=0.8), fontsize=10)

        # Marker untuk lokasi klik
        self.marker = None

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        print(f"Image size: {self.width} x {self.height}")
        print("Klik pada gambar untuk melihat info pixel")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Get pixel coordinates
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)

        # Check bounds
        if 0 <= row < self.height and 0 <= col < self.width:
            # Get NDVI value
            ndvi_value = float(self.ndvi_data[row, col])

            # Get geographic coordinates
            lon, lat = rasterio.transform.xy(self.src.transform, row, col)

            # Update info text
            info = f'Pixel: [{row}, {col}]\n'
            info += f'Lon: {lon:.6f}\n'
            info += f'Lat: {lat:.6f}\n'
            info += f'NDVI: {ndvi_value:.4f}'
            self.info_text.set_text(info)

            # Update marker
            if self.marker:
                self.marker.remove()
            self.marker = Circle((col, row), radius=5, color='red', fill=False, linewidth=2)
            self.ax.add_patch(self.marker)

            # Print to console
            print(f"\nPixel: [{row}, {col}]")
            print(f"Longitude: {lon:.6f}")
            print(f"Latitude: {lat:.6f}")
            print(f"NDVI: {ndvi_value:.4f}")

            self.fig.canvas.draw()

    def show(self):
        plt.show()

    def __del__(self):
        if hasattr(self, 'src'):
            self.src.close()


if __name__ == '__main__':
    tif_file = '/home/adedi/Documents/Tugas_Akhir/Data/Jember/Data/sawah3_clipped.tif'

    viewer = NDVIViewer(tif_file)
    viewer.show()
