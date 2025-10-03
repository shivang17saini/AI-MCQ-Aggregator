import os
import re
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from io import BytesIO

# OCR Imports
from PIL import Image
import pytesseract

# --- AI SDK Imports ---
import google.generativeai as genai
from google.api_core import exceptions
from mistralai.client import MistralClient
from huggingface_hub import InferenceClient
import openai # Used for OpenRouter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MODEL_TIMEOUT = 45

# --- API Client Initialization ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)
        logger.info("Google Gemini API configured successfully")
    else:
        logger.warning("GOOGLE_API_KEY not found")
except Exception as e:
    logger.error(f"Failed to configure Google Gemini API: {e}")

mistral_client = None
try:
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        mistral_client = MistralClient(api_key=mistral_api_key)
        logger.info("Mistral API configured successfully")
    else:
        logger.warning("MISTRAL_API_KEY not found")
except Exception as e:
    logger.warning(f"Mistral not configured: {e}")

hf_client = None
try:
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_api_key:
        hf_client = InferenceClient(token=hf_api_key)
        logger.info("Hugging Face API configured successfully")
    else:
        logger.warning("HUGGINGFACE_API_KEY not found")
except Exception as e:
    logger.warning(f"Hugging Face not configured: {e}")

openrouter_client = None
try:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        logger.info("OpenRouter API configured successfully")
    else:
        logger.warning("OPENROUTER_API_KEY not found")
except Exception as e:
    logger.warning(f"OpenRouter not configured: {e}")


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_text_input(text):
    if not text or not text.strip():
        return False, "Text input is empty"
    if len(text) > 5000:
        return False, "Text input is too long (max 5000 characters)"
    return True, None

def validate_image(image_data):
    if not image_data:
        return False, "No image data provided"
    try:
        image = Image.open(image_data)
        image.verify()
        image_data.seek(0)
        return True, None
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

# --- AI Query Functions ---
def query_gemini_model(prompt, model_name, model_display_name):
    """Generic function to query a Google Gemini model with retry logic."""
    logger.info(f"Starting Gemini query with model: {model_name}")
    if not os.getenv("GOOGLE_API_KEY"):
        return f"Error with {model_display_name}: API key not configured"
    
    try:
        model = genai.GenerativeModel(model_name)
        
        for attempt in range(3):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=800,
                        temperature=0.7,
                    ),
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                if hasattr(response, 'text') and response.text:
                    logger.info(f"{model_display_name} query successful")
                    return response.text
                elif hasattr(response, 'parts') and response.parts:
                    text = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
                    if text:
                        return text
                
                if hasattr(response, 'prompt_feedback'):
                    block_reason = response.prompt_feedback.block_reason if hasattr(response.prompt_feedback, 'block_reason') else 'UNKNOWN'
                    logger.warning(f"{model_display_name} blocked. Reason: {block_reason}")
                    return f"Response blocked by safety filters. Try rephrasing."
                
                return f"No response generated"

            except exceptions.ResourceExhausted as e:
                if attempt < 2:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"{model_display_name} rate limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"{model_display_name} rate limit after retries")
                    return f"Rate limit exceeded. Try again later."
            
            except Exception as e:
                logger.error(f"{model_display_name} error (attempt {attempt+1}): {e}")
                if attempt == 2:
                    return f"Error: {str(e)}"
                time.sleep(2)
        
        return "All retries failed"
        
    except Exception as e:
        logger.error(f"Failed to initialize {model_display_name}: {e}")
        return f"Initialization failed: {str(e)}"

def query_mistral(prompt):
    """Query Mistral Small"""
    try:
        logger.info("Starting Mistral query")
        if not mistral_client:
            return "API key not configured"
        
        response = mistral_client.chat(
            model='mistral-small-latest',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        logger.info("Mistral query completed")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Mistral error: {e}")
        return f"Error: {str(e)}"

def query_huggingface(prompt):
    """Query Hugging Face Inference API"""
    try:
        logger.info("Starting Hugging Face query")
        if not hf_client:
            return "API key not configured"
        
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_tokens=800,
            temperature=0.7,
        )
        logger.info("Hugging Face query completed")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Hugging Face error: {e}")
        return f"Error: {str(e)}"

def query_openrouter(prompt, model_id, display_name):
    """Generic function to query any OpenRouter model."""
    try:
        logger.info(f"Starting OpenRouter query for {display_name}")
        if not openrouter_client:
            return "OpenRouter API key not configured"
        
        response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        logger.info(f"OpenRouter query for {display_name} completed")
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OpenRouter error for {display_name}: {error_msg}")
        
        if "404" in error_msg or "not found" in error_msg.lower():
            return "Model not available"
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return "Invalid API key"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "Rate limit exceeded"
        else:
            return f"Error: {error_msg}"


# --- OCR Function ---
def get_text_from_image(image_data):
    try:
        logger.info("Starting OCR on image")
        image = Image.open(image_data)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        if not text or not text.strip():
            logger.warning("OCR returned empty text")
            return None
        logger.info(f"OCR completed, extracted {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        return None

def parse_answer(response_text):
    """Parses multiple questions and their answers from a response."""
    if not response_text or response_text.startswith("Error") or response_text.startswith("API"):
        return []

    question_answer_pattern = re.compile(
        r"(?:question|q)\s*(\d+)\s*[:\-\s]*\s*([A-D])", re.IGNORECASE
    )
    simple_pattern = re.compile(r"^\s*(\d+)\s*[\.\)]\s*([A-D])\b", re.IGNORECASE | re.MULTILINE)
    final_answer_pattern = re.compile(r"correct answer is\s*\*\*([A-D])\*\*", re.IGNORECASE)

    found_answers = {}

    for match in question_answer_pattern.finditer(response_text):
        q_num = int(match.group(1))
        answer = match.group(2).upper()
        if q_num not in found_answers:
            found_answers[q_num] = answer

    for match in simple_pattern.finditer(response_text):
        q_num = int(match.group(1))
        answer = match.group(2).upper()
        if q_num not in found_answers:
            found_answers[q_num] = answer
            
    if not found_answers:
        single_answer_match = re.search(
            r"(?:answer|option|choice)[\s:]*\(?([A-D])\)?", response_text, re.IGNORECASE
        )
        if single_answer_match:
            found_answers[1] = single_answer_match.group(1).upper()
        else:
            final_match = final_answer_pattern.search(response_text)
            if final_match:
                found_answers[1] = final_match.group(1).upper()

    return [{"question": q, "answer": a} for q, a in found_answers.items()]


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    available_models = []
    if os.getenv("GOOGLE_API_KEY"):
        available_models.append("Google Gemini")
    if mistral_client:
        available_models.append("Mistral")
    if hf_client:
        available_models.append("Hugging Face")
    if openrouter_client:
        available_models.append("OpenRouter")
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": available_models
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        logger.info("=== New Analysis Request ===")
        prompt_text = ""
        
        # Check for base64 image data (from paste)
        if 'image_data' in request.form and request.form['image_data']:
            try:
                import base64
                image_data = request.form['image_data']
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_stream = BytesIO(image_bytes)
                
                is_valid, error_msg = validate_image(image_stream)
                if not is_valid:
                    return jsonify({"error": error_msg}), 400
                    
                prompt_text = get_text_from_image(image_stream)
                if not prompt_text:
                    return jsonify({"error": "Could not read text from image."}), 400
            except Exception as e:
                logger.error(f"Error processing base64 image: {e}")
                return jsonify({"error": "Invalid image data"}), 400
        
        # Check for uploaded file
        elif 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            is_valid, error_msg = validate_image(image_file.stream)
            if not is_valid:
                return jsonify({"error": error_msg}), 400
            image_file.stream.seek(0)
            prompt_text = get_text_from_image(image_file.stream)
            if not prompt_text:
                return jsonify({"error": "Could not read text from image."}), 400
        
        # Check for text input
        elif 'text' in request.form and request.form['text'].strip() != '':
            prompt_text = request.form['text'].strip()
            is_valid, error_msg = validate_text_input(prompt_text)
            if not is_valid:
                return jsonify({"error": error_msg}), 400
        else:
            return jsonify({"error": "No input provided."}), 400

        final_prompt = (
            "You are an expert at solving multiple-choice questions. "
            "Analyze the following question(s) carefully and determine the correct answer. "
            "Provide a clear, step-by-step explanation. "
            "At the end, list the final answers clearly (e.g., '1. A', '2. C').\n\n"
            f"Question(s):\n{prompt_text}"
        )
        
        def generate_responses():
            queries = {}
            
            # --- Gemini Models (ONLY 2.0 FLASH) ---
            if os.getenv("GOOGLE_API_KEY"):
                queries["Google Gemini 2.0 Flash"] = lambda p=final_prompt: query_gemini_model(p, "gemini-2.0-flash-exp", "Google Gemini 2.0 Flash")

            # --- OpenRouter Models (WORKING FREE MODELS ONLY) ---
            if openrouter_client:
                openrouter_models = {
                    "Meta Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
                    "Qwen 2.5 72B": "qwen/qwen-2.5-72b-instruct:free",
                }
                for display_name, model_id in openrouter_models.items():
                    queries[display_name] = lambda p=final_prompt, m=model_id, d=display_name: query_openrouter(p, m, d)

            if not queries:
                error_data = {"type": "error", "data": {"message": "No AI models configured."}}
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                future_to_model = {executor.submit(query_func): model_name for model_name, query_func in queries.items()}
                all_responses = []
                
                for future in future_to_model:
                    model_name = future_to_model[future]
                    try:
                        start_time = time.time()
                        answer_text = future.result(timeout=MODEL_TIMEOUT)
                        duration = time.time() - start_time
                        
                        parsed_answers = parse_answer(answer_text)
                        result = {
                            "model": model_name, 
                            "answer": answer_text, 
                            "duration": duration, 
                            "parsed_answers": parsed_answers
                        }
                        all_responses.append(result)
                        
                        data = json.dumps({
                            "type": "model_result", 
                            "data": {
                                "model": model_name, 
                                "answer": answer_text, 
                                "duration": duration
                            }
                        })
                        yield f"data: {data}\n\n"

                    except FuturesTimeoutError:
                        result = {
                            "model": model_name, 
                            "answer": f"Request timed out after {MODEL_TIMEOUT} seconds", 
                            "parsed_answers": []
                        }
                        all_responses.append(result)
                        data = json.dumps({"type": "model_result", "data": result})
                        yield f"data: {data}\n\n"
                    except Exception as e:
                        result = {
                            "model": model_name, 
                            "answer": f"Error: {str(e)}", 
                            "parsed_answers": []
                        }
                        all_responses.append(result)
                        data = json.dumps({"type": "model_result", "data": result})
                        yield f"data: {data}\n\n"
            
            # --- Calculate Consensus ---
            question_votes = {}
            for res in all_responses:
                for parsed in res["parsed_answers"]:
                    q_num = parsed["question"]
                    ans = parsed["answer"]
                    if q_num not in question_votes:
                        question_votes[q_num] = {}
                    question_votes[q_num][ans] = question_votes[q_num].get(ans, 0) + 1

            consensus_data = []
            for q_num, votes in sorted(question_votes.items()):
                total_votes = sum(votes.values())
                percentages = {option: f"{(count / total_votes) * 100:.0f}%" for option, count in votes.items()}
                consensus_data.append({"question": q_num, "votes": percentages})

            data = json.dumps({"type": "consensus_result", "data": consensus_data})
            yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(generate_responses(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# --- Error Handlers ---
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File is too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error. Please try again later."}), 500

# --- Main ---
if __name__ == '__main__':
    logger.info("Starting AI MCQ Aggregator")
    logger.info(f"Model timeout: {MODEL_TIMEOUT} seconds")
    
    logger.info("\n=== Available AI Models ===")
    if os.getenv("GOOGLE_API_KEY"):
        logger.info("[OK] Google Gemini")
    if openrouter_client:
        logger.info("[OK] OpenRouter")
    logger.info("=" * 30 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)