from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient  # Import MongoDB connector
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  # Fixed: Correct import statement
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True #Add this to configure the output

# API KEYS
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Database config
MONGO_URI = os.getenv("MONGO_URI")  # Replace with your actual URI
DATABASE_NAME = "chemistry_db"
COLLECTION_NAME = "vector_store"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
vector_collection = db[COLLECTION_NAME]

# Function to get embeddings
def get_embeddings(text):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None


# Cosine similarity search function
def search_relevant_chunk(query, top_k=3):
    try:
        query_embedding = np.array(get_embeddings(query)).reshape(1, -1)

        stored_chunks = list(
            vector_collection.find({}, {"text": 1, "embeddings": 1, "_id": 0})
        )

        if not stored_chunks:
            return "No data found in the database."

        # Extract texts and embeddings
        texts = [chunk["text"] for chunk in stored_chunks]
        embeddings = np.array(
            [chunk["embeddings"] for chunk in stored_chunks], dtype=np.float32
        )  # 🔥 FIXED

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(texts[i], similarities[i]) for i in top_indices]

        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []


@app.route("/generate-test", methods=["POST"])
def generate_test():
    data = request.get_json()
    course = data["course"]
    num_questions = data["number_of_questions"]



    # 1. Retrieve Relevant Chunks (RAG)
    relevant_chunks = search_relevant_chunk(course)  # Use the course as the query

    if not relevant_chunks:
        return jsonify({"error": "No relevant course material found."}), 400

    # 2. Construct Enhanced Prompt
    context = "\n".join([chunk[0] for chunk in relevant_chunks])  # Get text values from chunk
    prompt = f"""
    You are an AI chemistry tutor. 
    A student is studying {course} and needs {num_questions} multiple-choice questions.

    Here is some relevant course material:
    {context}

    Generate {num_questions} multiple-choice questions based on the provided material, including the correct answer and a detailed explanation for each question.
    Your response must be valid JSON string and be careful to not contain any extra comma and adhere to coding standards.
    """
    

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    try:
        json_string = response.text.replace("```json", "").replace("```", "").strip()
        print(json_string)
        questions_json = json.loads(json_string)  # Parse the JSON string
        return jsonify(questions_json)  # Pass the parsed JSON data to jsonify, and configure output earlier

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return (
            jsonify({"error": "Failed to parse JSON from model response."}),
            500,
        )  # Return code, the return should be tuple

@app.route('/enquiry-room', methods=['POST'])
def enquiry_room():
    data = request.get_json()
    question_id = data['question_id']
    message = data['message']

    # 1. Retrieve Question Context (Optional)
    # You might want to retrieve the original question and explanation
    # from your database to provide the model with more context.

    # 2. Construct Prompt
    prompt = f"""
    You are an AI chemistry tutor. A student is asking about question ID {question_id}.

    Student's question: {message}

    Answer the student's question clearly and concisely.
    """

    # 3. Call Gemini Model
    model = genai.GenerativeModel("gemini-2.0-flash")  # Or your fine-tuned model name
    response = model.generate_content(prompt)

    # 4. Format and Return Response
    return jsonify({"response": response.text})

@app.route('/general-chat', methods=['POST'])
def general_chat():
    data = request.get_json()
    message = data['message']

    prompt = f"""
    You are an AI chemistry tutor acting as a study buddy. 
    A student has sent you the following message: "{message}"

    Your goals are to:
    1.  Provide a friendly and supportive response to the student.
    2.  Offer assistance with chemistry concepts, problem-solving, or study strategies, if relevant to the message.
    3.  Encourage the student to ask further questions or seek help when needed.

    Keep your responses concise, engaging, and appropriate for a study buddy relationship.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)