from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Tüm istemcilerden gelen istekleri kabul et

# Ortam değişkeninden model adı oku (bulamazsa default)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# SentenceTransformer modelini yükle
model = SentenceTransformer(MODEL_NAME)



@app.route('/', methods=['GET'])
def index():
    """Root endpoint to confirm the API is running."""
    # Return a simple JSON response to indicate that the server is running
    return jsonify({"message": "Hello, world! The API is running."})


@app.route('/api/v10/llm/vector', methods=['POST'])
def generate_vector():
    """Endpoint to convert provided text into a vector."""
    try:
        # Extract JSON data from the incoming request
        data = request.get_json()

        # Check if the data is present and contains the 'text' key
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided in the request body."}), 400
    
        # Retrieve the text to be vectorized
        text = data['text']

        # Validate that the provided text is not empty after stripping whitespace
        if not text.strip():
            return jsonify({"error": "The provided text is empty."}), 400

        # Encode the text using the SentenceTransformer model (CPU) and convert it to a list
        vector = model.encode(text).tolist()

        # Return the generated vector as a JSON response
        return jsonify({"vector": vector})

    except Exception as e:
        # Handle any unexpected errors and return a descriptive message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Ortam değişkenlerinden konfigürasyon al
    env = os.getenv("FLASK_ENV", "production")
    port = int(os.getenv("FLASK_PORT", "5003"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # Yalnızca development ortamında Flask dev server başlat
    if env == "development":
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            threaded=True
        )