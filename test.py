import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def preprocess_image_for_ocr(image_path, target_dpi=300, max_size=1800):
    """
    Hyper-tuned image preprocessing that adapts to different conditions:
    - Blur detection and correction
    - Lighting variation handling
    - Noise level assessment
    - Contrast enhancement
    - Shadow/uneven illumination correction
    """
    
    # Step 1: Set DPI and resize
    temp_path = set_image_dpi(image_path, target_dpi, max_size)
    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('processed_output2.jpg', img)
    print(f"Preprocessed image saved to: processed_output.jpg")
    # Step 2: Analyze image conditions
    conditions = analyze_image_conditions(img)
    print(f"Image conditions detected: {conditions}")

    
   
    
    # Step 4: Robust deskewing
    deskewed = robust_deskew(img)
    cv2.imwrite('processed_output4.jpg', deskewed)
    
    # Step 5: Adaptive thresholding with dynamic parameters
    final_img = dynamic_thresholding(deskewed, conditions)
    cv2.imwrite('processed_output5.jpg', final_img)
    
    
    # Step 7: Save preprocessed image
    cv2.imwrite('processed_output.jpg', final_img)
    print(f"Preprocessed image saved to: processed_output.jpg")
    
    return final_img

def analyze_image_conditions(img):
    """
    Analyze various image conditions to determine optimal preprocessing approach
    """
    conditions = {}
    
    # 1. Blur Detection using Laplacian variance
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    conditions['blur_level'] = 'high' if laplacian_var < 100 else 'medium' if laplacian_var < 500 else 'low'
    
    # 2. Lighting Analysis
    mean_brightness = np.mean(img)
    std_brightness = np.std(img)
    
    if mean_brightness < 80:
        conditions['lighting'] = 'dark'
    elif mean_brightness > 180:
        conditions['lighting'] = 'bright'
    else:
        conditions['lighting'] = 'normal'
    
    # 3. Contrast Analysis
    if std_brightness < 30:
        conditions['contrast'] = 'low'
    elif std_brightness > 80:
        conditions['contrast'] = 'high'
    else:
        conditions['contrast'] = 'normal'
    
    # 4. Noise Level Detection
    noise_level = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    noise_diff = np.mean(np.abs(img.astype(float) - noise_level.astype(float)))
    conditions['noise'] = 'high' if noise_diff > 15 else 'medium' if noise_diff > 5 else 'low'
    
    # 5. Shadow/Uneven Illumination Detection
    # Create a heavily blurred version to detect illumination patterns
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=img.shape[1]/30)
    illumination_var = np.std(blurred)
    conditions['illumination'] = 'uneven' if illumination_var > 20 else 'even'
    
    return conditions

def adaptive_preprocessing(img, conditions):
    """
    Apply preprocessing based on detected conditions
    """
    processed = img.copy()
    
    # Handle uneven illumination first
    if conditions['illumination'] == 'uneven':
        processed = correct_illumination(processed)
    
    # Handle lighting conditions
    if conditions['lighting'] == 'dark':
        processed = enhance_dark_image(processed)
    elif conditions['lighting'] == 'bright':
        processed = reduce_brightness(processed)
    
    # Handle contrast issues
    if conditions['contrast'] == 'low':
        processed = enhance_contrast(processed)
    elif conditions['contrast'] == 'high':
        processed = reduce_contrast(processed)
    
    # Handle noise based on level
    if conditions['noise'] == 'high':
        processed = cv2.fastNlMeansDenoising(processed, None, h=20, templateWindowSize=7, searchWindowSize=21)
    elif conditions['noise'] == 'medium':
        processed = cv2.fastNlMeansDenoising(processed, None, h=15, templateWindowSize=7, searchWindowSize=21)
    elif conditions['noise'] == 'low':
        processed = cv2.fastNlMeansDenoising(processed, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Handle blur
    if conditions['blur_level'] == 'high':
        processed = sharpen_image(processed)
    elif conditions['blur_level'] == 'medium':
        processed = mild_sharpen(processed)
    
    return processed

def correct_illumination(img):
    """
    Correct uneven illumination using background estimation
    """
    # Estimate background using morphological opening with large kernel
    kernel_size = max(img.shape[0], img.shape[1]) // 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Normalize by subtracting background
    corrected = cv2.subtract(img, background)
    corrected = cv2.add(corrected, np.median(img))
    
    return corrected

def enhance_dark_image(img):
    """
    Enhance dark/underexposed images
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Gamma correction for dark images
    gamma = 0.7
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    return enhanced

def reduce_brightness(img):
    """
    Reduce brightness for overexposed images
    """
    # Gamma correction for bright images
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    reduced = cv2.LUT(img, table)
    
    return reduced

def enhance_contrast(img):
    """
    Enhance contrast for low-contrast images
    """
    # CLAHE with higher clip limit for low contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    return enhanced

def reduce_contrast(img):
    """
    Reduce contrast for high-contrast images
    """
    # Apply mild Gaussian blur to reduce harsh contrasts
    reduced = cv2.GaussianBlur(img, (3, 3), 0)
    # Blend with original
    return cv2.addWeighted(img, 0.7, reduced, 0.3, 0)

def sharpen_image(img):
    """
    Sharpen blurry images using unsharp masking
    """
    # Create unsharp mask
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return unsharp_mask

def mild_sharpen(img):
    """
    Apply mild sharpening for moderately blurry images
    """
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)

def robust_deskew(image):
    """
    Enhanced deskewing with better angle detection
    """
    # Use Hough Line Transform for better angle detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:  # Only consider reasonable text angles
                angles.append(angle)
        
        if angles:
            # Use median angle for robustness
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:  # Only rotate if significant skew
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
    
    return image

def dynamic_thresholding(img, conditions):
    """
    Apply adaptive thresholding with dynamic parameters based on conditions
    """
    # Determine optimal blockSize based on image conditions
    if conditions['noise'] == 'high':
        block_size = 51  # Larger block for noisy images
    elif conditions['contrast'] == 'low':
        block_size = 31  # Medium block for low contrast
    else:
        block_size = 35  # Default
    
    # Determine C value based on lighting and contrast
    if conditions['lighting'] == 'dark':
        C = 15  # Higher C for dark images
    elif conditions['contrast'] == 'low':
        C = 5   # Lower C for low contrast
    else:
        C = 10  # Default
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )
    
    # For very low contrast images, try OTSU as fallback
    if conditions['contrast'] == 'low':
        _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine adaptive and OTSU
        thresh = cv2.bitwise_or(thresh, otsu_thresh)
    
    return thresh

def adaptive_morphology(img, conditions):
    """
    Apply morphological operations based on image conditions
    """
    # Determine kernel size based on noise and blur
    if conditions['noise'] == 'high' or conditions['blur_level'] == 'high':
        kernel_size = (3, 3)
        iterations = 2
    else:
        kernel_size = (2, 2)
        iterations = 1
    
    kernel = np.ones(kernel_size, np.uint8)
    
    # Opening to remove noise
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing to connect text components
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return closing

# Helper function (you'll need to implement this based on your earlier code)
def set_image_dpi(image_path, target_dpi=300, max_size=1800):
    """
    Set image DPI and resize (from your previous implementation)
    """
    im = Image.open(image_path)
    
    # Handle RGBA mode
    if im.mode == 'RGBA':
        background = Image.new('RGB', im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[-1])
        im = background
    elif im.mode not in ('RGB', 'L'):
        im = im.convert('RGB')
    
    length_x, width_y = im.size
    factor = max(1, int(max_size / length_x))
    size = factor * length_x, factor * width_y
    im_resized = im.resize(size, Image.LANCZOS)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(target_dpi, target_dpi))
    return temp_filename


filepath = '/home/priynashu/Pictures/Screenshots/Screenshot_20250722_165059.png'
processed_image = preprocess_image_for_ocr(filepath)

image = cv2.imread(processed_image)
cv2.imwrite('intermediae_outpt.jpg', image)
