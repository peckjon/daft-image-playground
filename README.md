# Daft Image Search Tool

An intelligent image search application that uses AI to automatically tag and make images searchable. Built with Daft.ai for efficient data processing, Flask for the backend, and a modern web interface.

## Features

### 🔄 Data Loader
- **Folder Processing**: Select any local folder containing images
- **Recursive Discovery**: Automatically finds images in all subfolders
- **AI-Powered Tagging**: Uses BLIP model for automatic image captioning and tagging
- **Batch Processing**: Efficiently processes large image collections using Daft.ai
- **Progress Tracking**: Real-time updates on processing status

### 🔍 Image Library
- **Smart Search**: Search images using natural language descriptions
- **Tag-Based Filtering**: Find images by automatically generated tags
- **Visual Preview**: Grid view with hover effects and click-to-expand
- **Detailed View**: Modal with full image, caption, tags, and metadata
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (for AI model)
- Modern web browser

### Installation

1. **Clone or download this repository**
2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Start the application:**
   ```bash
   source venv/bin/activate
   python app.py
   ```

4. **Open your browser to:** http://localhost:8000

## Usage Guide

### Processing Images

1. **Go to the Data Loader page**
2. **Enter your image folder path** (e.g., `/Users/yourname/Pictures`)
3. **Click "Start Processing"**
4. **Wait for completion** - the first run will download the AI model

**Example folder paths:**
- macOS: `/Users/yourname/Pictures/vacation`
- Linux: `/home/yourname/photos`
- Windows: `C:\\Users\\yourname\\Pictures`

### Searching Images

1. **Go to the Image Library page**
2. **Enter search terms** like:
   - "dog" (finds images with dogs)
   - "outdoor" (finds outdoor scenes)
   - "person" (finds images with people)
   - "mountain landscape" (finds mountain landscapes)
3. **Click on images** to see full size with details

## Technical Architecture

### Backend (Flask)
- **REST API** for image processing and search
- **Job management** for long-running processing tasks
- **File serving** for processed images

### Data Pipeline (Daft.ai)
- **Efficient file discovery** using glob patterns
- **Parallel processing** for image operations
- **Memory-efficient** handling of large datasets

### AI Processing
- **BLIP Model**: Salesforce's BLIP for image captioning
- **Automatic Tagging**: Extracts objects and scenes from captions
- **Standardized Output**: Consistent 224x224 processed images

### Data Storage
```json
{
  "images": [
    {
      "id": "abc123",
      "filename": "photo.jpg",
      "original_path": "/full/path/photo.jpg",
      "processed_path": "photo_abc123.jpg",
      "file_size": 1024576,
      "created_date": "2025-07-14T10:30:00Z",
      "tags": ["outdoor", "landscape", "mountains"],
      "caption": "A beautiful mountain landscape",
      "processed_date": "2025-07-14T15:45:00Z"
    }
  ]
}
```

## Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- BMP
- WEBP
- TIFF

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/process` | Start image processing job |
| GET | `/api/jobs/{id}` | Get processing job status |
| POST | `/api/search` | Search images by text |
| GET | `/api/images` | Get all processed images |

## File Structure

```
daft-image-playground/
├── app.py                 # Flask application
├── image_processor.py     # Core processing logic
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── templates/            # HTML templates
│   ├── data_loader.html
│   └── image_library.html
├── data/                 # Generated data files
├── processed_images/     # Resized images
└── README.md
```

## Performance Notes

- **First Run**: Model download may take 2-5 minutes
- **Processing Speed**: ~1-5 images per second (depending on hardware)
- **Memory Usage**: ~2-4GB during processing
- **Storage**: Processed images are ~50KB each (224x224 JPEG)

## Troubleshooting

### Common Issues

**"Model download failed"**
- Ensure internet connection for first run
- Check disk space (model is ~1GB)

**"Permission denied"**
- Ensure image folder is readable
- Use absolute paths, not relative

**"Out of memory"**
- Process smaller batches
- Close other applications
- Consider upgrading RAM

**"No images found"**
- Check folder path is correct
- Ensure folder contains supported image formats
- Verify folder permissions

### Logs and Debugging

- Processing logs appear in the terminal
- Check browser console for frontend errors
- Job status API provides detailed error messages

## Development

### Adding New Features

**Custom Image Models:**
- Replace BLIP model in `image_processor.py`
- Modify tag generation logic

**Search Improvements:**
- Add vector similarity search
- Implement advanced filtering

**UI Enhancements:**
- Add sorting options
- Implement image collections

### Dependencies

- **Daft.ai**: Distributed data processing
- **Flask**: Web framework
- **Transformers**: Hugging Face model library
- **Bootstrap**: UI framework

## License

This project is open source. Feel free to use and modify as needed.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Need Help?** Check the troubleshooting section or create an issue on GitHub.
