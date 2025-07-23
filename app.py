import streamlit as st
from paddleocr import PaddleOCR
from imageProcessing import process_image_complete
from textProcessing import EnhancedBusinessCardParser
from PIL import Image
import tempfile
import gc

# Streamlit App Config
st.set_page_config(
    page_title="Business Card OCR",
    layout="wide"
)

# Title
st.title("📇 Business Card OCR")
st.write("Capture or upload a business card to extract Name, Email, and Phone.")

# Initialize PaddleOCR once
@st.cache_resource
def get_ocr_instance():
    st.info("⏳ Loading OCR models... please wait")
    return PaddleOCR(
        lang='en',
        use_textline_orientation=True,
    )

ocr = get_ocr_instance()
parser = EnhancedBusinessCardParser()

# Dynamic camera size based on device
if st.session_state.get('mobile_view', False):
    camera_width = 400  # Mobile
    camera_height = 500
else:
    camera_width = 640  # Desktop
    camera_height = 480

# Toggle for mobile view
mobile_view = st.checkbox("📱 Mobile View (bigger camera)", value=False, key='mobile_view')

# Camera input
captured_image = st.camera_input(
    "📸 Capture Business Card",
    key="camera_input",
    help="Use your device's camera to take a picture of the business card.",
)

# OR Upload
uploaded_file = st.file_uploader(
    "📁 Or Upload an Image",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False
)

# Select input source
input_image = None
if captured_image is not None:
    input_image = Image.open(captured_image)
elif uploaded_file is not None:
    input_image = Image.open(uploaded_file)

# Process button
if input_image:
    st.image(input_image, caption="Uploaded Image", use_column_width=True)
    if st.button("🔍 Process Image"):
        # Save image to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            input_image.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Run OCR
            with st.spinner("Running OCR..."):
                ocr_result = process_image_complete(temp_file_path, ocr)

                if not ocr_result['success']:
                    st.error(f"OCR failed: {ocr_result['error']}")
                else:
                    # Parse results
                    parsed_data = parser.extract_info(ocr_result['extracted_texts'])
                    st.success("✅ Text extraction successful!")
                    st.json(parsed_data)

        except Exception as e:
            st.error(f"Error during OCR: {str(e)}")

        finally:
            # Clean up temp file
            gc.collect()
            temp_file.close()

else:
    st.info("📌 Capture or upload an image to start OCR.")

