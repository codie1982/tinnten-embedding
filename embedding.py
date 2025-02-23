from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

# Create the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Tüm istemcilerden gelen istekleri kabul et

# Load the SentenceTransformer model for encoding text into vectors (using CPU)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


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


# Run the Flask app, listening on all available IP addresses (host) at port 5003 with multithreading enabled
if __name__ == '__main__':
    # Enable multithreading and debug mode for concurrent request handling and easier troubleshooting
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
