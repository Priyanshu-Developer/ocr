import cv2
import numpy as np
from PIL import Image
import re
import gc
def set_image_dpi(file_path, target_dpi=300, max_size=1800):
    """
    Load image, set DPI, resize if needed, return NumPy array (BGR for OpenCV),
    and free any intermediate arrays to save memory.
    """
    im = Image.open(file_path)

    # Convert RGBA to RGB if necessary
    if im.mode == 'RGBA':
        background = Image.new('RGB', im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[-1])
        im = background
    elif im.mode not in ('RGB', 'L'):
        im = im.convert('RGB')

    # Resize if needed
    length_x, width_y = im.size
    factor = max(1, int(max_size / length_x))
    if factor > 1:
        size = factor * length_x, factor * width_y
        im = im.resize(size, Image.LANCZOS)

    # Convert to NumPy (RGB)
    np_image = np.array(im)

    # Convert RGB to BGR (for OpenCV) and make a safe copy
    del im  # Free PIL image memory
    np_image = np_image[:, :, ::-1].copy()

    gc.collect()  # Optional: force memory cleanup

    return np_image


def preprocess_image_for_ocr(image_path, target_dpi=300, max_size=1800):
    """
    Fully in-memory preprocessing pipeline. Returns a NumPy array ready for OCR.
    Cleans up intermediates to minimize memory usage.
    """
    # Step 1: Set DPI and resize
    img = set_image_dpi(image_path, target_dpi, max_size)

    # Step 2: Light denoising (preserve edges)
    denoised = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)
    del img  # Free memory
    gc.collect()

    # Step 3: Deskew image
    if len(denoised.shape) == 3:
        denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    def robust_deskew(image):
        _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 10 and w / h < 10:  # Text-like regions
                rect = cv2.minAreaRect(cnt)
                angle = rect[-1]
                if angle < -45:
                    angle += 90
                angles.append(angle)

        del th, contours  # Free intermediate threshold & contours
        gc.collect()

        if angles:
            median_angle = np.median(angles)
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            del M  # Free rotation matrix
            gc.collect()
            return rotated
        return image

    deskewed = robust_deskew(denoised)
    del denoised  # Free memory
    gc.collect()

    # Step 4: Adaptive thresholding (tuned)
    thresh = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,  # Larger block size
        C=10           # Adjusted for text clarity
    )
    del deskewed  # Free memory
    gc.collect()

    # Step 5: Morphological clean up
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    del thresh, kernel  # Free memory
    gc.collect()

    # Return final processed image as NumPy array
    return clean

def process_image_complete(image_path,ocr_instance):
    """
    Complete image processing pipeline for OCR.
    Optimized for speed and memory efficiency.
    """
    try:
        # Step 1: Preprocess the image
        processed_image = preprocess_image_for_ocr(image_path)
        cv2.imwrite('processed.png', processed_image)  # Save processed image if needed

        # Step 2: Ensure image has 3 channels (required by PaddleOCR)
        if processed_image.ndim == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        # Step 3: Get OCR instance (reused across calls)
        ocr = ocr_instance

        # Step 4: Run OCR inference
        results = ocr.predict(processed_image)

        # Optional: Cleanup unused variables to free memory
        del processed_image
        gc.collect()

   
        return {
            'success': True,
            'extracted_texts': [res['rec_texts'] for res in results][0],
        }
    except Exception as e:
        return {
            'success': False,
            'error': 'Error processing image: {}'.format(str(e))
        }