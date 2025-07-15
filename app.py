from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import uuid
from datetime import datetime
import threading
import daft
from image_processor import ImageProcessor
import shutil

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
    
    with open(json_path, 'r') as f:
        return json.load(f)

@app.route('/api/search', methods=['POST'])
def search_images():
    """Search for images based on text query using Daft"""
    query = request.json.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    data = load_processed_images()
    if not data:
        return jsonify({'results': [], 'message': 'No processed images found'})
    
    # Create Daft DataFrame from processed images
    df = daft.from_pylist(data.get('images', []))
    
    if len(df) == 0:
        return jsonify({'results': [], 'message': 'No processed images found'})
    
    query_lower = query.lower()
    
    # Collect all matching IDs from different searches
    matching_ids = set()
    
    # Search in tags - explode tags array and filter
    df_tags = df.explode(daft.col("tags"))
    df_tag_matches = df_tags.where(
        daft.col("tags").str.lower().str.contains(query_lower)
    ).collect()
    # Get IDs from tag matches
    if len(df_tag_matches) > 0:
        tag_ids = df_tag_matches.select("id").to_pylist()
        matching_ids.update([row["id"] for row in tag_ids])
    
    # Search in filenames
    df_filename_matches = df.where(
        daft.col("filename").str.lower().str.contains(query_lower)
    ).collect()
    # Get IDs from filename matches
    if len(df_filename_matches) > 0:
        filename_ids = df_filename_matches.select("id").to_pylist()
        matching_ids.update([row["id"] for row in filename_ids])
    
    # Search in captions
    df_caption_matches = df.where(
        daft.col("caption").str.lower().str.contains(query_lower)
    ).collect()
    # Get IDs from caption matches
    if len(df_caption_matches) > 0:
        caption_ids = df_caption_matches.select("id").to_pylist()
        matching_ids.update([row["id"] for row in caption_ids])
    
    # Filter original DataFrame to get all matches with consistent schema
    if matching_ids:
        all_matches = df.where(daft.col("id").is_in(list(matching_ids)))
    else:
        # Create empty DataFrame with same schema
        all_matches = df.where(daft.lit(False))
    
    # Convert to Python objects for JSON serialization
    results = all_matches.to_pylist()
    
    return jsonify({
        'results': results,
        'total': len(results),
        'query': query
    })

@app.route('/api/images')
def list_images():
    """Get all processed images using Daft"""
    data = load_processed_images()
    if not data:
        return jsonify({'images': [], 'total': 0})
    
    # Use Daft to handle the data
    df = daft.from_pylist(data.get('images', []))
    
    # Sort by processed_date (most recent first)
    df = df.sort(daft.col("processed_date"), desc=True)
    
    # Convert to Python objects for JSON serialization
    images = df.to_pylist()
    return jsonify({'images': images, 'total': len(images)})

@app.route('/processed_images/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    return send_from_directory(app.config['PROCESSED_IMAGES'], filename)

@app.route('/api/reset', methods=['POST'])
def reset_library():
    """Reset the image library by clearing all processed data"""
    # Remove JSON file
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_images.json')
    os.remove(json_path) if os.path.exists(json_path) else None
    
    # Clear processed images directory
    processed_dir = app.config['PROCESSED_IMAGES']
    shutil.rmtree(processed_dir, ignore_errors=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    return jsonify({'message': 'Image library reset successfully'})

if __name__ == '__main__':
    print("✅ Server starting at http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)