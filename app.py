import re
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# PostgreSQL connection details
DB_HOST = "dpg-d1vnkrndiees73brp680-a.oregon-postgres.render.com"
DB_PORT = 5432
DB_NAME = "client_jo5r"
DB_USER = "priyanshu"
DB_PASSWORD = "fw0lwMwJpbDYuTW9rwlBHB8w2HLAVoK8"  # <-- Change this

# SQLAlchemy Engine
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Initialize PaddleOCR
@st.cache_resource
def load_ocr():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

ocr = load_ocr()

# Common job-related keywords to exclude from names
job_keywords = {'EXECUTIVE', 'MANAGER', 'ENGINEER', 'DIRECTOR', 'CEO',
                'FOUNDER', 'OWNER', 'DEVELOPER', 'SECURITY', 'TECHNICIAN', 'STAFF', 'PVT', 'LTD', 'SERVICES'}

# Database functions
def insert_into_db(data):
    try:
        with engine.begin() as conn:
            conn.execute("""
                INSERT INTO business_cards (name, email, phone1, phone2, website, address)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                data["name"],
                data["email"],
                data["phone_numbers"][0] if len(data["phone_numbers"]) > 0 else None,
                data["phone_numbers"][1] if len(data["phone_numbers"]) > 1 else None,
                data["website"],
                data["address"]
            ))
        return True
    except Exception as e:
        st.error(f"❌ Database Error (insert): {e}")
        return False

def fetch_all_records():
    try:
        df = pd.read_sql("SELECT * FROM business_cards ORDER BY id DESC", engine)
        return df
    except Exception as e:
        st.error(f"❌ Could not fetch records: {e}")
        return pd.DataFrame()

# Extraction functions (PaddleOCR-based)
def extract_text(image_array):
    results = ocr.predict(image_array)
    text_lines = []
    for result in results:
        text_lines.extend(result['rec_texts'])
    return text_lines

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_phone(text):
    return re.findall(r'\+?\d[\d\s\-]{7,}\d', text)


def extract_name(text_lines):
    for line in text_lines:
        words = line.split()
        if (
            2 <= len(words) <= 4 and
            all(w.isalpha() for w in words) and
            not any(w.upper() in job_keywords for w in words)
        ):
            return line  # Return the first valid name
    return None



# Streamlit UI
st.set_page_config(page_title="📇 Business Card Extractor", layout="wide")
st.title("📇 Business Card Extractor (PaddleOCR) + PostgreSQL Viewer")

# Add custom CSS for larger camera canvas on mobile
st.markdown("""
    <style>
        [data-testid="stCameraInput"] video {
            width: 100% !important;
            height: auto !important;
        }
        [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Upload & Extract", "📑 View Saved Data"])

with tab1:
    st.header("Capture Business Card")
    camera_image = st.camera_input("Take a photo of the business card")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="📸 Captured Business Card", use_container_width=True)

        with st.spinner("🔍 Extracting data..."):
            text_lines = extract_text(np.array(image))
            full_text = "\n".join(text_lines)

            name = extract_name(text_lines)
            email = extract_email(full_text)
            phones = extract_phone(full_text)
           

            data = {
                "name": name,
                "email": email,
                "phone_numbers": phones,
                
            }

        st.success("✅ Data Extracted:")
        st.write(pd.DataFrame([data]))

        if st.button("💾 Save to Database"):
            if insert_into_db(data):
                st.success("🎉 Data saved successfully!")

with tab2:
    st.header("Saved Business Cards")
    records_df = fetch_all_records()
    if not records_df.empty:
        st.dataframe(records_df, use_container_width=True)
    else:
        st.info("No records found. Upload and save a card first.")
