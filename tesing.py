import cv2
import numpy as np
from PIL import Image
import pytesseract
import tempfile

def preprocess_image_for_ocr(image_path, target_dpi=300, max_size=1800):
    """
    Comprehensive image preprocessing for optimal Tesseract OCR results
    """
    
    # Step 1: Set optimal DPI and resize
    def set_image_dpi(file_path):
        im = Image.open(file_path)
        
        # Convert RGBA to RGB if necessary
        if im.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])  # Use alpha channel as mask
            im = background
        elif im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        
        length_x, width_y = im.size
        factor = max(1, int(max_size / length_x))
        size = factor * length_x, factor * width_y
        im_resized = im.resize(size, Image.LANCZOS)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(target_dpi, target_dpi))
        return temp_filename

    
    # Step 2: Load and convert to grayscale
    temp_path = set_image_dpi(image_path)
    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 3: Normalization
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    # Step 4: Noise removal using bilateral filter
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Step 5: Deskewing correction
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    deskewed = deskew(denoised)
    
    # Step 6: Adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(deskewed, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 2)
    
    # Step 7: Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    
    # Remove small noise - opening
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Close gaps in text - closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 8: Final smoothing
    def image_smoothening(img):
        ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3
    
    smoothed = image_smoothening(closing)
    
    # Step 9: Combine results using bitwise OR for best features
    final_image = cv2.bitwise_or(smoothed, closing)
    
    return final_image

def extract_text_with_confidence(processed_image):
    """
    Extract text using Tesseract with optimized settings
    """
    # Convert numpy array to PIL Image for Tesseract
    pil_image = Image.fromarray(processed_image)
    
    # Tesseract configuration for better accuracy
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-'
    
    # Extract text with confidence scores
    data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Filter results by confidence threshold
    min_confidence = 30
    filtered_text = []
    
    for i, conf in enumerate(data['conf']):
        if int(conf) > min_confidence:
            text = data['text'][i].strip()
            if text:
                filtered_text.append(text)
    
    return ' '.join(filtered_text)

# Main processing function
def process_image_complete(image_path):
    """
    Complete pipeline: preprocess image and extract text
    """
    try:
        # Preprocess the image
        processed_img = preprocess_image_for_ocr(image_path)
        
        # Save processed image for verification
        cv2.imwrite('processed_output.jpg', processed_img)
        
        # Extract text
        extracted_text = extract_text_with_confidence(processed_img)
        
        return {
            'success': True,
            'text': extracted_text,
            'processed_image_path': 'processed_output.jpg'
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Usage example
if __name__ == "__main__":
    result = process_image_complete('/home/priynashu/Desktop/ocr/frontend/public/captures/capture-1753242186707.png')
    
    if result['success']:
        print("Extracted Text:")
        print(result['text'])
        print(f"\nProcessed image saved as: {result['processed_image_path']}")
    else:
        print(f"Error: {result['error']}")
