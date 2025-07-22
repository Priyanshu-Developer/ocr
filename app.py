import streamlit as st
import easyocr
import re
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

# Initialize EasyOCR reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()



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

# Extraction Functions
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_phone(text):
    return re.findall(r'\+?\d[\d\s\-]{7,}\d', text)

def extract_website(text):
    match = re.search(r'((http(s)?://)?[\w\-]+\.\w{2,}(\.\w{2,})?)', text)
    return match.group(0) if match else None

def extract_name(text_lines):
    for line in text_lines:
        if not any(char.isdigit() for char in line) and len(line.split()) >= 2:
            return line
    return None

def extract_address(text_lines):
    possible = []
    for line in reversed(text_lines):
        if not re.search(r'@|www|\.com|\d{5,}', line):
            possible.append(line)
        if len(possible) >= 2:
            break
    return ", ".join(reversed(possible))

# Streamlit UI
st.title("📇 Business Card Extractor + PostgreSQL Viewer")


tab1, tab2 = st.tabs(["📤 Upload & Extract", "📑 View Saved Data"])

with tab1:
    st.header("Upload a Business Card")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Business Card", use_container_width=True)

        with st.spinner("🔍 Extracting data..."):
            results = reader.readtext(np.array(image), detail=0)
            full_text = "\n".join(results)

            name = extract_name(results)
            email = extract_email(full_text)
            phones = extract_phone(full_text)
            website = extract_website(full_text)
            address = extract_address(results)

            data = {
                "name": name,
                "email": email,
                "phone_numbers": phones,
                "website": website,
                "address": address
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
