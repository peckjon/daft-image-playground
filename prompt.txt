# Daft Image Search Tool - Product Requirements Document

## Project Overview
We're creating an intelligent image search tool using Daft.ai for efficient data processing, Flask for the backend API, and jQuery for the frontend interface.

## Technical Stack
- **Backend**: Flask (Python)
- **Frontend**: jQuery, Bootstrap, HTML5
- **Data Pipeline**: Daft.ai for distributed processing
- **AI Models**: BLIP (Salesforce) for image captioning and tagging
- **Image Processing**: Pillow for resizing and format handling

## Core Features

### Data Loader Page
**Purpose**: Process and catalog images from local folders

**Features**:
- Allow users to specify local folder paths for image processing
- Recursively discover all images in folders and subfolders
- Use Daft.ai to efficiently batch process large image collections
- Resize all images to standardized dimensions (224x224) for consistency
- Run each image through BLIP AI model for automatic captioning
- Extract relevant tags from generated captions and filenames
- Generate structured JSON database with image metadata
- Provide real-time progress tracking during processing
- Handle multiple image formats (JPEG, PNG, GIF, WEBP, BMP, TIFF)

**Technical Requirements**:
- Parallel processing using Daft.ai DataFrames
- Background job management with status tracking
- Error handling for corrupted or unsupported files
- Memory-efficient processing for large datasets
- Comprehensive metadata extraction (file size, dates, dimensions)

### Image Library Page
**Purpose**: Search and browse processed image collections

**Features**:
- Text-based search using natural language queries
- Tag-based filtering and discovery
- Responsive grid layout for image browsing
- Modal view for detailed image inspection
- Search result statistics and sorting
- Mobile-friendly responsive design
- Fast search performance with client-side optimization

**Technical Requirements**:
- RESTful API for search operations
- Efficient JSON-based data querying
- Lazy loading for performance optimization
- Semantic search matching tags and captions
- Progressive image loading with thumbnails

## Data Schema
```json
{
  "metadata": {
    "processed_date": "ISO timestamp",
    "source_folder": "original folder path",
    "total_images": "number of processed images"
  },
  "images": [
    {
      "id": "unique identifier",
      "filename": "original filename",
      "original_path": "full path to original",
      "processed_path": "path to resized image",
      "file_size": "size in bytes",
      "created_date": "file creation date",
      "tags": ["array", "of", "extracted", "tags"],
      "caption": "AI-generated description",
      "processed_date": "processing timestamp"
    }
  ]
}
```

## API Endpoints
- `POST /api/process` - Start image processing job
- `GET /api/jobs/{id}` - Check processing status
- `POST /api/search` - Search images by text query
- `GET /api/images` - Retrieve all processed images
- `GET /processed_images/{filename}` - Serve processed image files

## User Experience Goals
- **Simplicity**: Intuitive interface requiring minimal user training
- **Performance**: Fast processing and search response times
- **Reliability**: Robust error handling and progress feedback
- **Scalability**: Handle large image collections efficiently
- **Accessibility**: Responsive design working across devices

## Success Metrics
- Process 1000+ images in under 5 minutes
- Search response time under 2 seconds
- Support for all common image formats
- 90%+ accuracy in automatic tagging
- Mobile-responsive interface