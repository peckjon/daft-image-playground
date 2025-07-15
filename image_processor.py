import daft
import os
import json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import requests

class ImageProcessor:
    def __init__(self):
        # Initialize the image captioning model
        self.processor = None
        self.model = None
        self.stopwords = set()
        self.load_stopwords()
        self.load_model()
        
    def load_stopwords(self):
        """Load English stopwords from a remote source or use fallback"""
        try:
            # Try to load from local cache first
            cache_file = 'stopwords_cache.txt'
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self.stopwords = set(word.strip().lower() for word in f.readlines() if word.strip())
                print(f"Loaded {len(self.stopwords)} stopwords from cache")
                return
            
            # Download from a reliable source
            print("Downloading English stopwords...")
            url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse stopwords
            stopwords_list = [word.strip().lower() for word in response.text.split('\n') if word.strip()]
            self.stopwords = set(stopwords_list)
            
            # Cache for future use
            with open(cache_file, 'w') as f:
                for word in sorted(self.stopwords):
                    f.write(f"{word}\n")
            
            print(f"Downloaded and cached {len(self.stopwords)} stopwords")
            
        except Exception as e:
            print(f"Failed to download stopwords: {e}")
            print("Using minimal fallback stopwords")
            # Minimal fallback stopwords
            self.stopwords = {
                'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
        
    def load_model(self):
        """Load the BLIP model for image captioning"""
        try:
            print("Loading BLIP model for image recognition...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing without AI model - will use basic filename tags")
    
    def find_images(self, folder_path):
        """Use Daft to find all images in folder and subfolders"""
        # Create a Daft DataFrame from the file system
        df = daft.from_glob_path(f"{folder_path}/**/*")
        
        # Filter for image files
        df = df.where(
            df["path"].str.endswith(".jpg") |
            df["path"].str.endswith(".jpeg") |
            df["path"].str.endswith(".png") |
            df["path"].str.endswith(".gif") |
            df["path"].str.endswith(".bmp") |
            df["path"].str.endswith(".webp") |
            df["path"].str.endswith(".tiff")
        )
        
        return df
    
    def convert_uri_to_path(self, uri_path):
        """Convert file:// URI to local file path"""
        if uri_path.startswith('file://'):
            parsed = urlparse(uri_path)
            return parsed.path
        return uri_path
    
    def resize_image_udf(self, image_path, target_size=(224, 224)):
        """UDF function to resize a single image for Daft bulk processing"""
        try:
            # Convert URI to path if needed
            local_path = self.convert_uri_to_path(image_path)
            
            with Image.open(local_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize with aspect ratio preservation
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create a new image with the target size and paste the resized image
                new_img = Image.new('RGB', target_size, (255, 255, 255))
                
                # Calculate position to center the image
                x = (target_size[0] - img.width) // 2
                y = (target_size[1] - img.height) // 2
                
                new_img.paste(img, (x, y))
                
                # Generate output filename
                filename = os.path.basename(local_path)
                name, ext = os.path.splitext(filename)
                image_hash = hashlib.md5(local_path.encode()).hexdigest()[:8]
                processed_filename = f"{name}_{image_hash}.jpg"
                
                # Create output directory if it doesn't exist
                output_dir = 'processed_images'
                os.makedirs(output_dir, exist_ok=True)
                processed_path = os.path.join(output_dir, processed_filename)
                
                # Save resized image
                new_img.save(processed_path, 'JPEG', quality=85)
                
                # Get file stats
                file_stats = os.stat(local_path)
                
                return {
                    'original_path': local_path,
                    'processed_filename': processed_filename,
                    'processed_path': processed_path,
                    'filename': filename,
                    'file_size': file_stats.st_size,
                    'created_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'image_hash': image_hash,
                    'success': True,
                    'error': None
                }
                
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return {
                'original_path': image_path,
                'processed_filename': None,
                'processed_path': None,
                'filename': os.path.basename(image_path) if isinstance(image_path, str) else 'unknown',
                'file_size': 0,
                'created_date': datetime.now().isoformat(),
                'image_hash': None,
                'success': False,
                'error': str(e)
            }
    
    def generate_tags_and_caption(self, image_path):
        """Generate tags and caption for an image using BLIP model"""
        if self.model is None or self.processor is None:
            # Fallback to basic tags if model is not loaded
            return ["image"], "An image file"
        
        try:
            # Load and process image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Generate caption
                inputs = self.processor(img, return_tensors="pt")
                out = self.model.generate(**inputs, max_length=50)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                # Split caption into words and clean them
                words = caption.lower().split()
                tags = []
                
                for word in words:
                    # Remove punctuation from word
                    clean_word = ''.join(char for char in word if char.isalnum())
                    
                    # Skip if word is too short, too long, or in stopwords
                    if (len(clean_word) >= 3 and 
                        len(clean_word) <= 20 and 
                        clean_word not in self.stopwords and
                        clean_word.isalpha()):  # Only alphabetic characters
                        tags.append(clean_word)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_tags = []
                for tag in tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                
                # Add some basic tags if none found
                if not unique_tags:
                    unique_tags = ['image']
                
                return unique_tags, caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return ["image"], "An image file"
    
    def process_single_image(self, image_path, output_dir):
        """Process a single image: resize, generate caption and tags"""
        try:
            # Generate unique filename for processed image
            image_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            processed_filename = f"{name}_{image_hash}.jpg"
            processed_path = os.path.join(output_dir, processed_filename)
            
            # Resize image
            resized_image = self.resize_image(image_path)
            if resized_image is None:
                return None
            
            # Save resized image
            resized_image.save(processed_path, 'JPEG', quality=85)
            
            # Generate tags and caption
            tags, caption = self.generate_tags_and_caption(resized_image)
            
            # Get file stats
            file_stats = os.stat(image_path)
            
            return {
                'id': image_hash,
                'filename': filename,
                'original_path': image_path,
                'processed_path': processed_filename,
                'file_size': file_stats.st_size,
                'created_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'tags': tags,
                'caption': caption,
                'processed_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_folder(self, folder_path, job_id, processing_jobs):
        """Process all images in a folder using Daft for bulk operations"""
        try:
            # Update job status
            processing_jobs[job_id]['status'] = 'discovering'
            
            # Find all images using Daft
            print(f"Discovering images in {folder_path}...")
            df_images = self.find_images(folder_path)
            
            # Convert to Python list to get count
            image_files = df_images.collect()
            total_images = len(image_files)
            processing_jobs[job_id]['total_images'] = total_images
            
            if total_images == 0:
                processing_jobs[job_id]['status'] = 'completed'
                processing_jobs[job_id]['message'] = 'No images found in the specified folder'
                return
            
            print(f"Found {total_images} images to process")
            processing_jobs[job_id]['status'] = 'processing'
            
            # Create output directory
            output_dir = 'processed_images'
            os.makedirs(output_dir, exist_ok=True)
            
            # Use Daft to bulk resize images with built-in functions
            print("Bulk resizing images with Daft's built-in resizer...")
            
            # First read the files as binary data
            df_images = df_images.with_column(
                "file_data",
                daft.col("path").url.download()
            )
            
            # Then decode as images
            df_images = df_images.with_column(
                "image_data",
                daft.col("file_data").image.decode()
            )
            
            # Resize images using Daft's built-in resizer
            target_size = (224, 224)
            df_images = df_images.with_column(
                "resized_image",
                daft.col("image_data").image.resize(target_size[0], target_size[1])
            )
            
            # Save resized images and process tags/captions
            resized_results = df_images.collect()
            
            print("Generating AI tags and captions...")
            processed_images = []
            
            for i, row in enumerate(resized_results):
                try:
                    # Convert URI to local path
                    local_path = self.convert_uri_to_path(row['path'])
                    filename = os.path.basename(local_path)
                    name, ext = os.path.splitext(filename)
                    image_hash = hashlib.md5(local_path.encode()).hexdigest()[:8]
                    processed_filename = f"{name}_{image_hash}.jpg"
                    processed_path = os.path.join(output_dir, processed_filename)
                    
                    # Convert numpy array to PIL image
                    resized_array = row['resized_image']
                    if hasattr(resized_array, 'to_pil'):
                        # If it's a Daft image object
                        pil_image = resized_array.to_pil()
                    else:
                        # If it's a numpy array
                        import numpy as np
                        if isinstance(resized_array, np.ndarray):
                            # Convert numpy array to PIL Image
                            # Ensure the array is in the right format (uint8, RGB)
                            if resized_array.dtype != np.uint8:
                                resized_array = (resized_array * 255).astype(np.uint8)
                            pil_image = Image.fromarray(resized_array)
                        else:
                            # Fallback to original image processing
                            print(f"Unexpected image format for {filename}, falling back to PIL resize")
                            with Image.open(local_path) as img:
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                pil_image = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    pil_image.save(processed_path, 'JPEG', quality=85)
                    
                    # Generate tags and caption
                    tags, caption = self.generate_tags_and_caption(processed_path)
                    
                    # Get file stats
                    file_stats = os.stat(local_path)
                    
                    image_data = {
                        'id': image_hash,
                        'filename': filename,
                        'original_path': local_path,
                        'processed_path': processed_filename,
                        'file_size': file_stats.st_size,
                        'created_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'tags': tags,
                        'caption': caption,
                        'processed_date': datetime.now().isoformat()
                    }
                    processed_images.append(image_data)
                except Exception as e:
                    print(f"Skipping failed image: {row.get('path', 'unknown')} - {e}")
                
                # Update progress
                processing_jobs[job_id]['processed_images'] = i + 1
                processing_jobs[job_id]['progress'] = int((i + 1) / total_images * 100)
            
            # Save results to JSON
            output_data = {
                'metadata': {
                    'processed_date': datetime.now().isoformat(),
                    'source_folder': folder_path,
                    'total_images': len(processed_images),
                    'failed_images': total_images - len(processed_images)
                },
                'images': processed_images
            }
            
            json_path = os.path.join('data', 'processed_images.json')
            os.makedirs('data', exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Update final job status
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['end_time'] = datetime.now().isoformat()
            processing_jobs[job_id]['output_file'] = json_path
            processing_jobs[job_id]['successful_images'] = len(processed_images)
            
            print(f"Processing completed! Successfully processed {len(processed_images)} images")
            
        except Exception as e:
            print(f"Error in process_folder: {e}")
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = str(e)
