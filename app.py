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
    json_path = os.path.join('data', 'processed_images.json')
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)

@app.route('/api/search', methods=['POST'])
def search_images():
    """Search for images based on text query using Daft SQL"""
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
    
    # Explode tags for searching
    df_exploded = df.explode(daft.col("tags"))

    # Get IDs that match in both tags and filename
    combined_search_sql = f"""
    SELECT DISTINCT id
    FROM df_exploded
    WHERE LOWER(tags) LIKE '%{query_lower}%'
       OR LOWER(filename) LIKE '%{query_lower}%'
    """

    # Execute query to get matching IDs
    matching_ids_result = daft.sql(combined_search_sql).to_pylist()
    matching_ids = [row['id'] for row in matching_ids_result]
    
    # Filter original DataFrame to get results with proper tags array
    if matching_ids:
        results_df = df.where(daft.col("id").is_in(matching_ids))
        results = results_df.to_pylist()
    else:
        results = []
    
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
    return send_from_directory('processed_images', filename)

@app.route('/api/browse', methods=['POST'])
def browse_folders():
    """Browse folders starting from a given path"""
    try:
        data = request.get_json() or {}
        start_path = data.get('path')
        
        # Use home directory if no path provided or path is None
        if not start_path:
            start_path = os.path.expanduser('~')
        
        # Sanitize and validate path
        start_path = os.path.abspath(start_path)
        if not os.path.exists(start_path) or not os.path.isdir(start_path):
            start_path = os.path.expanduser('~')
        
        folders = []
        files = []
        
        try:
            for item in sorted(os.listdir(start_path)):
                if not item:  # Skip empty names
                    continue
                    
                item_path = os.path.join(start_path, item)
                
                # Skip hidden files and system folders
                if item.startswith('.'):
                    continue
                    
                try:
                    if os.path.isdir(item_path):
                        # Check if directory is accessible
                        try:
                            os.listdir(item_path)
                            folders.append({
                                'name': item,
                                'path': item_path,
                                'type': 'folder'
                            })
                        except (PermissionError, OSError):
                            # Skip inaccessible folders
                            continue
                    elif os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')):
                        files.append({
                            'name': item,
                            'path': item_path,
                            'type': 'image'
                        })
                except (OSError, IOError):
                    # Skip files that can't be accessed
                    continue
        except PermissionError:
            return jsonify({'error': 'Permission denied to access this folder'}), 403
        
        # Get parent directory safely
        parent_path = None
        if start_path != '/':  # Unix root
            parent_candidate = os.path.dirname(start_path)
            if parent_candidate != start_path:  # Avoid infinite loop on Windows drives
                parent_path = parent_candidate
        
        return jsonify({
            'current_path': start_path,
            'parent_path': parent_path,
            'folders': folders,
            'image_files': files,
            'total_images': len(files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_library():
    """Reset the image library by clearing all processed data"""
    # Remove JSON file
    json_path = os.path.join('data', 'processed_images.json')
    os.remove(json_path) if os.path.exists(json_path) else None
    
    # Clear processed images directory
    shutil.rmtree('processed_images', ignore_errors=True)
    os.makedirs('processed_images', exist_ok=True)
    
    return jsonify({'message': 'Image library reset successfully'})

if __name__ == '__main__':
    print("✅ Server starting at http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)