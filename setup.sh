#!/bin/bash

echo "🚀 Setting up Daft Image Search Tool..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the Flask app: python app.py"
echo "3. Open your browser to: http://localhost:8000"
echo ""
echo "📋 Usage:"
echo "• Data Loader: Process images from a local folder"
echo "• Image Library: Search through your processed images"
echo ""
echo "💡 Tips:"
echo "• On first run, the AI model will be downloaded (this may take a few minutes)"
echo "• Supported formats: JPG, JPEG, PNG, GIF, BMP, WEBP, TIFF"
echo "• For best results, ensure your image folder path is absolute"
