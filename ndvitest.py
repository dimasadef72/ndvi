import os
import sys
from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import piexif


os.environ['MPLCONFIGDIR'] = './.matplotlib_cache'
def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def compress_image(input_path, output_path, quality=20, resize_percentage=50):
    """
    Compresses the image by reducing its quality and resizing.
    Parameters:
    - input_path: path to the original image file
    - output_path: path to save the compressed image
    - quality: quality for JPEG compression (lower value means higher compression)
    - resize_percentage: percentage of the original size to resize (e.g., 50 means 50%)
    """
    try:
        with Image.open(input_path) as img:
            # Resize the image
            if resize_percentage < 100:
                new_width = int(img.width * resize_percentage / 100)
                new_height = int(img.height * resize_percentage / 100)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save with lower quality
            img.save(output_path, quality=quality, optimize=True)
        print(f"Image successfully compressed and resized to {resize_percentage}% as {output_path}")
    except Exception as e:
        print(f"Failed to compress image: {e}")

def compress_images_in_folder(input_folder, output_folder, file_suffix, quality=20, resize_percentage=50):
    ensure_folder_exists(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(file_suffix):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            compress_image(input_path, output_path, quality, resize_percentage)

def calculate_ndvi(nir_folder, red_folder, output_folder):
    ensure_folder_exists(output_folder)
    nir_files = [f for f in os.listdir(nir_folder) if f.endswith('_NIR.TIF')]
    red_files = [f for f in os.listdir(red_folder) if f.endswith('_R.TIF')]
    
    for nir_file in nir_files:
        base_name = nir_file.replace('_NIR.TIF', '')
        red_file = f'{base_name}_R.TIF'
        
        if red_file in red_files:
            nir_path = os.path.join(nir_folder, nir_file)
            red_path = os.path.join(red_folder, red_file)
            
            nir_image = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            red_image = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)
            
            if nir_image is not None and red_image is not None:
                # Compute NDVI
                nir = nir_image.astype(float)
                red = red_image.astype(float)
                ndvi = (nir - red) / (nir + red + 1e-10)
                
                # Normalize and apply colormap
                ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                colormap = cm.get_cmap('RdYlGn')
                ndvi_colored = (colormap(ndvi_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
                
                # Save NDVI image with EXIF data
                output_filename = f'{base_name}_NDVI_Colored.jpg'
                ndvi_output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(ndvi_output_path, cv2.cvtColor(ndvi_colored, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                
                # Add EXIF metadata from the NIR image
                exif_dict = piexif.load(nir_path)
                ndvi_image_pil = Image.open(ndvi_output_path)
                exif_bytes = piexif.dump(exif_dict)
                ndvi_image_pil.save(ndvi_output_path, "jpeg", exif=exif_bytes)
                
                print(f"NDVI image with EXIF saved for: {base_name}")
            else:
                print(f"Failed to load images for: {base_name}")
        else:
            print(f'No matching Red image for NIR image: {nir_file}')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images


def load_and_resize_images_from_folder(folder, scale_percent=30):
    # Memuat dan mengurangi resolusi gambar dari folder
    if not os.path.exists(folder):
        print("Folder does not exist!")
        return []
    
    images = []
    print("Processing files in folder:")
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and os.path.isfile(img_path):
            print(f"Processing file: {filename}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipped file (not a valid image): {filename}")
                continue
            
            # Mengubah resolusi gambar
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            
            images.append(resized_img)
        else:
            print(f"Skipped file (not an image): {filename}")
    return images

def check_image_format(images):
    # Memeriksa format gambar untuk memastikan validitas
    for img in images:
        if img is None or len(img.shape) != 3:
            print("Invalid image format or corrupted image detected.")
            return False
    return True

def stitch_multiple_images(image_folder, output_file, mode=cv2.Stitcher_SCANS, batch_size=20, scale_percent=30):
    # Memuat dan mengurangi resolusi gambar
    images = load_and_resize_images_from_folder(image_folder, scale_percent=scale_percent)
    
    if len(images) < 2:
        print("Tidak cukup gambar untuk melakukan stitching.")
        return
    
    # Memeriksa format gambar
    if not check_image_format(images):
        print("Invalid image format detected. Exiting process.")
        return
    
    # Menggunakan OpenCV's Stitcher untuk menjahit gambar
    stitcher = cv2.Stitcher_create(mode)
    
    # Memproses gambar dalam batch
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: {len(batch_images)} images")

        try:
            status, stitched_image =    stitcher.stitch(batch_images)

            if status == cv2.Stitcher_OK:
                # Menyimpan hasil stitching batch
                output_batch_file = f"{output_file[:-4]}_batch{i // batch_size + 1}.jpg"
                cv2.imwrite(output_batch_file, stitched_image)
                print(f"Gambar berhasil dijahit untuk batch {i // batch_size + 1} dan disimpan di {output_batch_file}")
            else:
                print(f"Gagal menjahit gambar di batch {i // batch_size + 1}. Status error: {status}")
        except cv2.error as e:
            print(f"Error saat stitching batch {i // batch_size + 1}: {e}")
            print(f"Skipping batch {i // batch_size + 1} dan lanjut ke batch berikutnya...")

def main():
    # Define paths for each data type
    rgb_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/rgb_data"
    nir_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/nir_data"
    red_folder = "/home/adedi/Documents/Tugas_Akhir/Data/Drone Mentah/DJI_202409071655_007_sawah1/red_data"
    output_folder = "./output_ndvi"

    # Define relative paths based on the output_folder
    compressed_rgb_folder = os.path.join(output_folder, "compressed_rgb")
    compressed_nir_red_folder = os.path.join(output_folder, "compressed_nir_red")
    ndvi_folder = os.path.join(output_folder, "ndvi_output")
    stitching_rgb_folder = os.path.join(output_folder, "stitching_rgb")
    stitching_ndvi_folder = os.path.join(output_folder, "stitching_ndvi")
    stitching_rgb_output_file = os.path.join(stitching_rgb_folder, "stitched_output_rgb.jpg")
    stitching_ndvi_output_file = os.path.join(stitching_ndvi_folder, "stitched_output_ndvi.jpg")

    # Ensure required folders exist
    ensure_folder_exists(stitching_rgb_folder)
    ensure_folder_exists(stitching_ndvi_folder)

    # Compress images with additional resizing
    compress_images_in_folder(rgb_folder, compressed_rgb_folder, '_D.JPG', quality=20, resize_percentage=50)
    compress_images_in_folder(nir_folder, compressed_nir_red_folder, '_NIR.TIF', quality=20, resize_percentage=50)
    compress_images_in_folder(red_folder, compressed_nir_red_folder, '_R.TIF', quality=20, resize_percentage=50)
    
    # Calculate NDVI
    calculate_ndvi(compressed_nir_red_folder, compressed_nir_red_folder, ndvi_folder)
    
    # Perform stitching on compressed RGB and NDVI images
    stitch_multiple_images(compressed_rgb_folder, stitching_rgb_output_file, mode=cv2.Stitcher_SCANS, batch_size=20, scale_percent=30)
    stitch_multiple_images(ndvi_folder, stitching_ndvi_output_file, mode=cv2.Stitcher_SCANS, batch_size=20, scale_percent=30)

if __name__ == "__main__":
    main()