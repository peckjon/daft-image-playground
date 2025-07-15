from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import uuid
from datetime import datetime
import threading
from image_processor import ImageProcessor

print("🚀 Starting Daft Image Search Tool...")

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'data',
    'PROCESSED_IMAGES': 'processed_images'
})

# Create directories
for folder in ['data', 'processed_images', 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

print("🤖 Initializing AI image processor...")
processing_jobs = {}
image_processor = ImageProcessor()

@app.route('/')
def index():
    return render_template('data_loader.html')

@app.route('/library')
def library():
    return render_template('image_library.html')

@app.route('/api/process', methods=['POST'])
def process_images():
    """Start processing images from a specified folder"""
    folder_path = request.json.get('folder_path')
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        'status': 'started',
        'progress': 0,
        'total_images': 0,
        'processed_images': 0,
        'start_time': datetime.now().isoformat(),
        'folder_path': folder_path
    }
    
    thread = threading.Thread(
        target=image_processor.process_folder,
        args=(folder_path, job_id, processing_jobs),
        daemon=True
    )
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'started'})

@app.route('/api/jobs/<job_id>')
def get_job_status(job_id):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(processing_jobs[job_id])

def load_processed_images():
    """Helper to load processed images JSON"""
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@app.route('/api/search', methods=['POST'])
def search_images():
    """Search for images based on text query"""
    query = request.json.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    data = load_processed_images()
    if not data:
        return jsonify({'results': [], 'message': 'No processed images found'})
    
    query_lower = query.lower()
    results = []
    
    for image_data in data.get('images', []):
        tags = image_data.get('tags', [])
        filename = image_data.get('filename', '')
        
        if (any(query_lower in tag.lower() for tag in tags) or 
            query_lower in filename.lower()):
            results.append(image_data)
    
    return jsonify({
        'results': results,
        'total': len(results),
        'query': query
    })

@app.route('/api/images')
def list_images():
    """Get all processed images"""
    data = load_processed_images()
    if not data:
        return jsonify({'images': [], 'total': 0})
    
    images = data.get('images', [])
    return jsonify({'images': images, 'total': len(images)})

@app.route('/processed_images/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    return send_from_directory(app.config['PROCESSED_IMAGES'], filename)

@app.route('/api/reset', methods=['POST'])
def reset_library():
    """Reset the image library by clearing all processed data"""
    try:
        # Remove JSON file
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
        if os.path.exists(json_path):
            os.remove(json_path)
        
        # Clear processed images
        processed_dir = app.config['PROCESSED_IMAGES']
        if os.path.exists(processed_dir):
            for filename in os.listdir(processed_dir):
                file_path = os.path.join(processed_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        return jsonify({'message': 'Image library reset successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to reset library: {str(e)}'}), 500

if __name__ == '__main__':
    print("✅ Server starting at http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)
