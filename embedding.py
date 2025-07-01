from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode
import asyncio
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, JsonXPathExtractionStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import CrawlerMonitor, DisplayMode, RateLimiter
import uuid
import json
import nest_asyncio
nest_asyncio.apply()
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

@app.route('/api/v10/llm/scrapper', methods=['POST'])
def run_scraper():
    try:
        data = request.get_json()
        url  = data.get('url', '').strip()

        if not url:
            return jsonify({"error": "No valid URL provided."}), 400

        # Async fonksiyonu çalıştır
        result = asyncio.run(extract_content_from_url(url))

        # Başarısızsa, kullanıcıya hata mesajı dön
        if result.get("success") is False:
            return jsonify({
                "error": result.get("message", "Unknown error"),
                "status_code": result.get("status_code", 500)
            }), result.get("status_code", 500)

        return jsonify({"crawler": result}), 200

    except Exception as e:
        print("❌ SCRAPER HATASI:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

async def extract_content_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True,
        headers=headers
    )

    config = CrawlerRunConfig(
        verbose=True,
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True
    )

    async with AsyncWebCrawler(config=browser_cfg, verbose=True) as crawler:
        result = await crawler.arun(url=url, config=config)

        if not result.success:
            print(f"❌ CRAWL FAILED for {result.url}: {result.status_code} - {result.error_message}")
            return {
                "success": False,
                "url": result.url,
                "status_code": result.status_code,
                "message": result.error_message or "Crawling failed"
            }

        product_info = {
            "success": True,
            "url": result.url,
            "data": result.extracted_content,
            "images": result.media.get("images", []),
            "metadata": result.metadata,
            "markdown_v2": result.markdown_v2.raw_markdown if result.markdown_v2 else ""
        }
        return product_info


  
# Run the Flask app, listening on all available IP addresses (host) at port 5003 with multithreading enabled
if __name__ == '__main__':
    # Enable multithreading and debug mode for concurrent request handling and easier troubleshooting
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)
