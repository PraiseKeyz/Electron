import os
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from pymongo import MongoClient
  
 # Download tokenizer

# Database setup
client = MongoClient("mongodb+srv://praiseoluwatobilobaadebayo:uN9ALRzFxQ8I61kQ@cluster0.jkfis.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["chemistry_db"]
collection = db["course_chunks"]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return None  # Return None so we can handle it in the main function

# Function to split text into chunks
def chunk_text(text, max_length=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Process all PDFs in multiple courses
def process_all_courses(base_folder):
    for course_folder in os.listdir(base_folder):
        course_path = os.path.join(base_folder, course_folder)

        if os.path.isdir(course_path):  # Ensure it's a folder
            print(f"📖 Processing course: {course_folder}")

            pdf_files = [f for f in os.listdir(course_path) if f.endswith(".pdf")]
            if not pdf_files:
                print(f"⚠️ No PDFs found in {course_folder}, skipping.")
                continue

            for pdf_file in pdf_files:
                pdf_path = os.path.join(course_path, pdf_file)
                text = extract_text_from_pdf(pdf_path)

                if text:  # Proceed only if text extraction was successful
                    chunks = chunk_text(text)

                    # Insert into MongoDB with unique constraint
                    for chunk in chunks:
                        collection.update_one(
                            {
                                "course_name": course_folder,
                                "pdf_name": pdf_file,
                                "text": chunk
                            },
                            {"$setOnInsert": {
                                "course_name": course_folder,
                                "pdf_name": pdf_file,
                                "text": chunk
                            }},
                            upsert=True
                        )

                    print(f"✅ Stored {len(chunks)} chunks from {pdf_file} in {course_folder}")

# Run the process
process_all_courses("Course_pdfs")
