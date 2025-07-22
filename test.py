import re
from paddleocr import PaddleOCR

import os
os.environ["PADDLE_MODEL_HOME"] = "./models"

# Initialize OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Path to local image
image_path = "/home/priynashu/Desktop/ocr/WhatsApp Image 2025-07-22 at 9.57.13 PM.jpeg"

# Run OCR inference
results = ocr.predict(input=image_path)

# Initialize containers for extracted data
emails = []
phones = []
possible_names = []

# Regex patterns
email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
phone_pattern = re.compile(r"(\+?\d[\d\s-]{8,}\d)")

# Common job-related keywords to exclude
job_keywords = {'EXECUTIVE', 'MANAGER', 'ENGINEER', 'DIRECTOR', 'CEO',
                'FOUNDER', 'OWNER', 'DEVELOPER', 'SECURITY', 'TECHNICIAN', 'STAFF', 'PVT', 'LTD', 'SERVICES'}

print("🔍 Detected Text:")
for result in results:
    rec_texts = result['rec_texts']
    rec_scores = result['rec_scores']

    for text, score in zip(rec_texts, rec_scores):
        print(f"{text} (Confidence: {score:.2f})")

        # Search for email
        found_emails = email_pattern.findall(text)
        emails.extend(found_emails)

        # Search for phone numbers
        found_phones = phone_pattern.findall(text)
        phones.extend(found_phones)

        # Heuristic for possible names
        words = text.split()
        if (
            2 <= len(words) <= 4 and                         # 2-4 words
            all(w.isalpha() for w in words) and              # Alphabetic only
            not any(w.upper() in job_keywords for w in words)  # No job keywords
        ):
            possible_names.append(text)

# Remove duplicates
emails = list(set(emails))
phones = list(set(phones))
possible_names = list(set(possible_names))

print("\n📧 Extracted Emails:")
for email in emails:
    print(f" - {email}")

print("\n📞 Extracted Phone Numbers:")
for phone in phones:
    print(f" - {phone}")

print("\n👤 Possible Names:")
for name in possible_names:
    print(f" - {name}")
