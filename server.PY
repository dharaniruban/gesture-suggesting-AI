from flask import Flask, request, render_template, send_from_directory
import os
import logging
from app import process_video  # Import process_video from app.py

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define and create uploads folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.logger.info(f"Uploads folder created/verified at: {UPLOAD_FOLDER}")
except Exception as e:
    app.logger.error(f"Failed to create uploads folder: {str(e)}")

# Log template folder
TEMPLATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.logger.info(f"Template folder set to: {TEMPLATE_FOLDER}")

# Route to serve index.html
@app.route('/')
def index():
    app.logger.info("Serving index.html")
    return render_template('index.html')

# Route to serve results.html
@app.route('/results.html')
def results():
    app.logger.info("Serving results.html")
    return render_template('results.html')

# Route to handle video analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            app.logger.error("No video file provided in request")
            return {"error": "No video file provided"}, 400
        video = request.files['video']
        if video.filename == '':
            app.logger.error("No video file selected")
            return {"error": "No video selected"}, 400

        # Save the video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        app.logger.info(f"Saving video to: {video_path}")
        video.save(video_path)

        # Verify file was saved
        if not os.path.exists(video_path):
            app.logger.error(f"Failed to save video to: {video_path}")
            return {"error": "Failed to save video file"}, 500

        # Process video using app.py
        app.logger.info(f"Processing video with app.py: {video_path}")
        result = process_video(video_path)
        app.logger.info(f"Processing result: {result}")

        # Include video filename in response
        result['videoFilename'] = video.filename
        return result
    except Exception as e:
        app.logger.error(f"Error in /analyze: {str(e)}")
        return {"error": f"Failed to process video: {str(e)}"}, 500

# Route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Route to serve uploaded videos
@app.route('/Uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)