from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import uuid
from datetime import datetime
import threading
from image_processor import ImageProcessor

print("🚀 Starting Daft Image Search Tool...")
print("📦 Initializing Flask application...")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['PROCESSED_IMAGES'] = 'processed_images'

print("📁 Creating necessary directories...")
# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_IMAGES'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("🤖 Initializing AI image processor (this may take a moment)...")
# Global variables for job tracking
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
    data = request.json
    folder_path = data.get('folder_path')
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = {
        'status': 'started',
        'progress': 0,
        'total_images': 0,
        'processed_images': 0,
        'start_time': datetime.now().isoformat(),
        'folder_path': folder_path
    }
    
    # Start processing in a separate thread
    thread = threading.Thread(
        target=image_processor.process_folder,
        args=(folder_path, job_id, processing_jobs)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'started'})

@app.route('/api/jobs/<job_id>')
def get_job_status(job_id):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_jobs[job_id])

@app.route('/api/search', methods=['POST'])
def search_images():
    """Search for images based on text query"""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Load the processed images data
    json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
    
    if not os.path.exists(json_file_path):
        return jsonify({'results': [], 'message': 'No processed images found'})
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Simple text search in tags and descriptions
        results = []
        query_lower = query.lower()
        
        for image_data in data.get('images', []):
            # Search in tags
            tags = image_data.get('tags', [])
            tag_match = any(query_lower in tag.lower() for tag in tags)
            
            # Search in filename
            filename_match = query_lower in image_data.get('filename', '').lower()
            
            if tag_match or filename_match:
                results.append(image_data)
        
        return jsonify({
            'results': results,
            'total': len(results),
            'query': query
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/images')
def list_images():
    """Get all processed images"""
    json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
    
    if not os.path.exists(json_file_path):
        return jsonify({'images': [], 'total': 0})
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        images = data.get('images', [])
        return jsonify({
            'images': images,
            'total': len(images)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load images: {str(e)}'}), 500

@app.route('/processed_images/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    return send_from_directory(app.config['PROCESSED_IMAGES'], filename)

@app.route('/api/reset', methods=['POST'])
def reset_library():
    """Reset the image library by clearing all processed data"""
    try:
        # Remove the JSON file
        json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        
        # Clear the processed images directory
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
    print("✅ Initialization complete!")
    print("🌐 Starting Flask web server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📱 Or access from other devices at: http://0.0.0.0:8000")
    print("🔧 Running in debug mode - changes will auto-reload")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=8000)
