<img src="https://github.com/user-attachments/assets/9513c930-fc68-4ae4-b5d7-d5acfe702784" width="1" height="1" style="display:none;" alt="Daft Image Playground">

<div align="center" style="max-width: 720px; margin: 0 auto;">
<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/71fce0b2-e37e-4a4d-8649-336ced5a42f6" type="video/mp4">
  Image Playground demo using Daft.ai
</video>
</div>

# Building an AI-Powered Image Search Engine with Daft.ai

Back when ML models mainly lived on self-hosted servers instead of smartphones, I spent a few years with [Algorithmia](https://github.com/algorithmiaio), building some of the first and best ML (now "AI") hosting services. Many of my days were spent deep in the trenches with Python datascientists, churning through Jupyter notebooks, optimizing their algorithms to run in ephemeral serverless environments. Those were the days when data transformation pipelines required complex orchestration of multiple tools, custom scripts for every file format, and endless debugging of memory issues and race conditions.

Fast-forward to today: after years focused on DevOps and other areas of software development, I've been itching to get back into data science – and wow, the modern landscape is a revelation. Enter [Daft](https://daft.ai): a distributed Python dataframe library designed to handle complex data workloads with the elegance of Pandas but the power to scale. What caught my attention wasn't just another dataframe library, but Daft's native support for multimodal data processing and SQL-based query capabilities. This felt like the perfect opportunity to build something practical while exploring what makes Daft exciting.

## Why Daft is Worth Your Attention

Daft represents a significant step forward in data processing, especially for teams working with unstructured data. Unlike traditional dataframes that treat multimedia as mere file paths, Daft can natively decode, process, and manipulate images directly within its processing pipeline. This means you can resize thousands of images, extract features, or run ML inference – all using familiar dataframe operations that can scale across multiple cores or even distributed clusters.

Structured data gets an upgrade, too! Daft's built-in support for SQL queries works across nonrelational data, such as JSON... so those of us who grew up writing SQL92 feel just as comfortable querying a wide variety of formats.

The three Daft features that really shine in this project are:

**🔍 [Image Discovery & File Processing](https://docs.getdaft.io/en/stable/api/io/#daft.from_glob_path)**: Using `daft.from_glob_path()`, we can recursively discover image files across directory structures with built-in filtering by extension. No more writing custom directory traversal code or managing file system complexity.

**⚡ [Bulk Image Processing](https://docs.getdaft.io/en/stable/api/expressions/#daft.expressions.expressions.ExpressionImageNamespace)**: Daft's native image operations let us chain `.image.decode()`, `.image.resize()`, and `.image.encode()` in a single pipeline. This means processing thousands of photos happens in parallel without having to manually manage Pillow operations, threading, or memory concerns.

**📊 [SQL Query over JSON](https://docs.getdaft.io/en/stable/sql_overview/)**: Once our image metadata is processed, Daft's SQL interface `daft.sql()` lets us write SQL queries directly over our JSON data structures, including complex operations that replace slow and cumbersome dataframe operations -- like array explosions for tag searching, and querying across multiple fields simultaneously.

## Building the Demo: Where Theory Meets Practice

This image search tool demonstrates how these capabilities come together. The application discovers images in local folders, processes them through AI models for automatic captioning and tagging, then creates a searchable web interface. Here's where Daft eliminated entire categories of complexity:

- **No manual file system traversal** – Daft's glob patterns handle recursive file discovery: [image_processor.py#L40](https://github.com/peckjon/daft-image-playground/blob/main/image_processor.py#L40)
- **No individual image resize operations** – Daft's bulk image pipeline processes everything in parallel (no sequential Pillow operations!): [image_processor.py#L135](https://github.com/peckjon/daft-image-playground/blob/main/image_processor.py#L135)
- **No complex JSON parsing for search** – SQL queries over structured data feel natural and performant: [app.py#L95-L106](https://github.com/peckjon/daft-image-playground/blob/main/app.py#L95-L106)
- **No manual parallelization** – Daft handles efficient resource utilization automatically: [image_processor.py#L132](https://github.com/peckjon/daft-image-playground/blob/main/image_processor.py#L132)

The result? Clean, readable code that focuses on business logic rather than infrastructure concerns.

## Development Notes & Caveats

Full transparency: while the initial code generation was aided by GitHub Copilot and Claude Sonnet 4 (you can see the original prompt in [PRD.md](PRD.md) – itself pair-generated with Copilot's help), the real work happened in the development iterations. AI tools are incredibly powerful accelerators, but they work best when guided by an experienced developer who understands the problem domain and can refine the generated solutions.

**Important**: This is a demo application only and should not be used unmodified in a production environment. It may contain security vulnerabilities and is optimized for simplicity and compatibility, not efficiency. For example, the BLIP model used for image captioning is a few years old and not state-of-the-art – I chose it for its reliability and broad compatibility rather than cutting-edge performance.

This project showcases only a tiny slice of [Daft's capabilities](https://daft.ai). The framework supports everything from distributed computing across cloud infrastructure to advanced ML workloads with GPU acceleration. If you're dealing with large-scale data processing, multimedia pipelines, or looking to modernize your data infrastructure, there's a lot more to explore.

## Ready to dive in?

🚀 [Jump right into the code](https://github.com/peckjon/daft-image-playground) or read the detailed implementation guide below!

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
   chmod +x setup.sh
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
      "created_date": "2025-07-14T10:30:00",
      "tags": ["outdoor", "landscape", "mountains"],
      "caption": "A beautiful mountain landscape",
      "processed_date": "2025-07-14T15:45:00"
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
├── LICENSE               # MIT License
├── templates/            # HTML templates
│   ├── data_loader.html
│   └── image_library.html
├── data/                 # Generated data files
├── processed_images/     # Resized images
└── README.md
```

## Performance Notes

- **First Run**: Model download may take 2-5 minutes
- **Processing Speed**: Varies by hardware and image size
- **Memory Usage**: ~2-4GB during processing
- **Storage**: Processed images are ~50KB each (224x224 JPEG)
- **Model Size**: ~1GB

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

### Interested in taking this further? A few suggestions:

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
- **Transformers**: Hugging Face model library
- **Flask**: Web framework
- **Bootstrap**: UI framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License allows you to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Place warranty
- ✅ Use patents

The only requirement is to include the original copyright notice.

---

**Need Help?** Check the troubleshooting section or create an issue on GitHub.
