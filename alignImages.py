import os
import cv2
import numpy as np
import imageio

def load_image(file_path):
    """Loads an image from the specified file path."""
    # Check the file extension to determine how to handle the file
    _, file_ext = os.path.splitext(file_path)
    if file_ext.lower() in ['.tiff', '.tif']:
        # Use imageio for TIFF files which might contain multiple layers or pages
        return imageio.volread(file_path)
    elif file_ext.lower() in ['.jpg', '.jpeg']:
        # Use OpenCV for JPEG files
        return cv2.imread(file_path, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Unsupported file format: {}".format(file_ext))

def align_images(input_files):
    # Load the base image (reference)
    base_image = load_image(input_files[0])
    if len(base_image.shape) == 3 and base_image.shape[2] == 3:
        base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    else:
        base_gray = base_image  # Assume already grayscale or single layer

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in base image
    keypoints_base, descriptors_base = sift.detectAndCompute(base_gray, None)

    # Initialize FLANN based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    aligned_images = [base_image]

    for file_path in input_files[1:]:
        current_image = load_image(file_path)
        if len(current_image.shape) == 3 and current_image.shape[2] == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_image

        keypoints_current, descriptors_current = sift.detectAndCompute(current_gray, None)

        matches = flann.knnMatch(descriptors_base, descriptors_current, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            aligned = cv2.warpPerspective(current_image, matrix, (base_image.shape[1], base_image.shape[0]))
            aligned_images.append(aligned)
        else:
            print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
            aligned_images.append(current_image)

    stacked_image = np.stack(aligned_images, axis=0)
    return stacked_image

def save_as_tiff(stacked_image, output_file):
    imageio.mimwrite(output_file, stacked_image, format='TIFF')

def save_as_envi(stacked_image, output_file):
    save_image(output_file, stacked_image, dtype='float32', force=True)

def main():
    input_folder = 'C:\\Users\\hdavies\\OneDrive - University of Colorado Colorado Springs\\Saltveit'
    output_folder = 'C:\\Users\\hdavies\\OneDrive - University of Colorado Colorado Springs\\Saltveit\\PythonTest'
    os.makedirs(output_folder, exist_ok=True)

    # Collect all JPEG and TIFF files from the input folder
    valid_extensions = ['.jpg', '.jpeg', '.tif', '.tiff']
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    if not input_files:
        print("No compatible image files found in the input folder.")
        return

    stacked_image = align_images(input_files)

    tiff_output_file = os.path.join(output_folder, 'aligned_images.tif')
    envi_output_file = os.path.join(output_folder, 'aligned_images.hdr')

    save_as_tiff(stacked_image, tiff_output_file)
    print(f"TIFF file saved as {tiff_output_file}")

    try:
        save_as_envi(stacked_image, envi_output_file)
        print(f"ENVI file saved as {envi_output_file}")
    except Exception as e:
        print(f"Could not save as ENVI: {e}")

if __name__ == "__main__":
    main()
