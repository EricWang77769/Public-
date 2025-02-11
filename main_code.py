import json
import os
from pathlib import Path
import cv2
import google.generativeai as genai
import numpy as np
import logging
import absl.logging
import firebase_admin
from firebase_admin import credentials, storage, db

# Initialize Firebase with your credentials and URLs
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'eric-farming-app.firebasestorage.app',
    'databaseURL': 'https://eric-farming-app-default-rtdb.firebaseio.com/'
})


bucket = storage.bucket()

# Initialize logging
absl.logging.set_verbosity(absl.logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from JSON file
with open("api_key.json") as f:
    api_key = json.load(f)["api_key"]

genai.configure(api_key=api_key)

def load_image(image_path):
    """Loads an image from file."""
    return cv2.imread(image_path)

def write_detections(image, object_info, raw_response=None):
    """Write detected object names and facts on the image."""
    height, width, _ = image.shape
    white_space_width = 400
    white_space = 255 * np.ones((height, white_space_width, 3), np.uint8)
    image = cv2.hconcat([image, white_space])

    font_scale = max(0.3, min(width, height) / 1500)  # Smaller scale for narrow images
    thickness = max(1, int(font_scale))
    y_position = 30
    line_spacing = int(font_scale * 10)  # Dynamic line spacing

    if raw_response:
        logger.warning("Writing raw response to image due to parsing errors.")
        clean_response = clean_raw_response(raw_response)
        for line in clean_response.splitlines():
            if y_position + 20 > height:
                logger.warning("Not enough space to write raw response on image.")
                break
            cv2.putText(image, line.strip(), (width + 10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
            y_position += 20
        return image

    for info in object_info:
        label = info.get("name", "Unknown")
        label = (label[:30] + '...') if len(label) > 30 else label
        fact = info.get("fact", "No details available")

        label_text = f'{label}'
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(image, label_text, (width + 10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        y_position += text_height + line_spacing

        max_line_width = white_space_width - 20
        words = fact.split(' ')
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            test_width, _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            if test_width > max_line_width:
                cv2.putText(image, current_line, (width + 10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                y_position += text_height + 5
                current_line = word + " "
            else:
                current_line = test_line

        if current_line:
            cv2.putText(image, current_line, (width + 10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
            y_position += text_height + line_spacing

        if y_position + text_height > height:
            logger.warning("Not enough space to draw all labels.")
            break

    return image

def clean_raw_response(raw_response):
    """Cleans up the raw response text for better readability."""
    lines = raw_response.splitlines()
    clean_lines = []
    for line in lines:
        if line.startswith("{") or line.startswith("}") or line.startswith("[") or line.startswith("]"):
            continue
        clean_lines.append(line.strip())
    return "\n".join(clean_lines)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return None

def create_gemini_model(model_name):
    """Create a Gemini model instance."""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 64,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain",
    }
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

def parse_raw_response(response_text):
    """Attempts to parse the raw response text into JSON."""
    try:
        # Extract JSON content if enclosed in triple backticks
        if "```json" in response_text:
            start = response_text.index("```json") + 7
            end = response_text.index("```", start)
            response_text = response_text[start:end].strip()
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return [{"name": "Raw Response", "fact": response_text.strip()}]


def format_json_for_output(raw_json):
    """Format the raw JSON output to match the desired structure."""
    formatted_json = []
    for item in raw_json:
        name = item.get("name", "Unknown")
        fact = item.get("fact", "No details available")
        if name != "Unknown" or fact != "No details available":
            formatted_json.append({"name": name, "fact": fact})
    return formatted_json

def get_object_info_from_image(image_path):
    """Get object information from the image using Gemini."""
    image_file = upload_to_gemini(image_path, mime_type="image/jpeg")
    if not image_file:
        return [{"name": "Unknown", "fact": "Could not retrieve fact"}], None

    model = create_gemini_model("gemini-1.5-flash")
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [image_file],
            },
        ]
    )

    response = chat_session.send_message(
        "Analyze the image in about 100 words or less. Focus on plants, pots, and any related details pertaining to plant care. Provide a JSON array with 'name' and 'fact' fields for each object."
    )

    response_text = response.text.strip()
    raw_json = parse_raw_response(response_text)

    if raw_json:
        formatted_data = format_json_for_output(raw_json)
        return formatted_data, None
    else:
        return [{"name": "Raw Response", "fact": clean_raw_response(response_text)}], None

def save_json_data(json_data, output_path):
    """Save JSON data to a text file."""
    try:
        with open(output_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        logger.info(f"Saved JSON data: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON data: {e}")

def upload_to_firebase_storage(local_path, remote_path):
    """Uploads a file to Firebase Storage."""
    try:
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        blob.make_public()  # Make the file publicly accessible
        logger.info(f"Uploaded {local_path} to Firebase Storage at {remote_path}")
        return blob.public_url
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to Firebase Storage: {e}")
        return None

def save_metadata_to_realtime_database(filename, image_url, json_url):
    """Save file metadata to Firebase Realtime Database."""
    try:
        ref = db.reference("files").push()  # Create a new entry in the 'files' node
        ref.set({
            "filename": filename,
            "image_url": image_url,
            "json_url": json_url
        })
        logger.info(f"Saved metadata to Realtime Database: {filename}")
    except Exception as e:
        logger.error(f"Failed to save metadata to Realtime Database: {e}")

if __name__ == "__main__":
    output_dir = Path("/home/pi/Output")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path("/home/pi/Plants")
    images = [file for file in image_dir.iterdir() if file.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    for image_file in images:
        filename = image_file.stem
        logger.info(f"Processing: {filename}")

        # Get object information from the image
        object_info, raw_response = get_object_info_from_image(image_file)
        logger.info(f"Detected objects: {object_info}")

        # Save the raw JSON data
        json_output_path = output_dir / f"{filename}_output.json"
        save_json_data(object_info, json_output_path)

        # Load and annotate the image
        image = load_image(str(image_file))
        if image is None:
            logger.error(f"Failed to load image: {image_file}")
            continue

        # Write detections to the image
        processed_image = write_detections(image, object_info, raw_response=None)

        # Save the processed image
        image_output_path = output_dir / f"{filename}_output.jpg"
        if not cv2.imwrite(str(image_output_path), processed_image):
            logger.error(f"Failed to save image at {image_output_path}")
            continue

        logger.info(f"Saved processed image: {image_output_path}")

        # Ask user if they want to upload to Firebase
        user_input = input(f"Do you want to upload '{filename}' to Firebase? (y/n): ").strip().lower()

        if user_input == 'y':
            # Upload to Firebase Storage
            firebase_image_path = f"images/{filename}_output.jpg"
            firebase_json_path = f"json/{filename}_output.json"

            image_url = upload_to_firebase_storage(str(image_output_path), firebase_image_path)
            json_url = upload_to_firebase_storage(str(json_output_path), firebase_json_path)

            if image_url and json_url:
                # Save metadata to Realtime Database
                save_metadata_to_realtime_database(filename, image_url, json_url)
                logger.info(f"Uploaded files to Firebase and metadata saved to Realtime Database.")
            else:
                logger.error("Failed to upload files to Firebase or save metadata.")
        else:
            logger.info(f"Skipped uploading '{filename}' to Firebase.")
