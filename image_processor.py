import daft
import os
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import numpy as np

class ImageProcessor:
    def __init__(self):
        # Initialize the image captioning model
        self.processor = None
        self.model = None
        self.stopwords = set()
        self.load_stopwords()
        self.load_model()
        
    def load_stopwords(self):
        """Load English stopwords from local file"""
        try:
            # Load from local stopwords file
            stopwords_file = 'stopwords-en.txt'
            with open(stopwords_file, 'r') as f:
                # Skip comment lines that start with #
                self.stopwords = set(
                    word.strip().lower() 
                    for word in f.readlines() 
                    if word.strip() and not word.strip().startswith('#')
                )
            print(f"Loaded {len(self.stopwords)} stopwords from {stopwords_file}")
        except Exception as e:
            print(f"Failed to load stopwords from file: {e}")
            self.stopwords = set()  # Empty set if file can't be loaded
        
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
    
    def generate_tags_and_caption_from_array(self, image_array):
        """Generate tags and caption for a numpy image array using BLIP model"""
        if self.model is None or self.processor is None:
            # Fallback to basic tags if model is not loaded
            return ["image"], "An image file"
        
        try:
            # Ensure we have a proper numpy array
            if not isinstance(image_array, np.ndarray):
                print(f"Expected numpy array, got {type(image_array)}")
                return ["image"], "An image file"
            
            # For BLIP, we need to create a tensor directly from numpy array
            # Convert numpy array to the format BLIP expects
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Convert to float and normalize to [0, 1] for BLIP
            image_tensor = torch.from_numpy(image_array).float() / 255.0
            
            # Rearrange dimensions from HWC to CHW for BLIP
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Normalize using BLIP's expected values (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Generate caption using tensor directly
            inputs = {"pixel_values": image_tensor}
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
            print(f"Error generating caption: {e}")
            return ["image"], "An image file"
    
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
            
            # Encode resized images back to JPEG bytes for saving
            df_images = df_images.with_column(
                "encoded_image",
                daft.col("resized_image").image.encode("JPEG")
            )
            
            # Collect results
            resized_results = df_images.collect()
            
            print("Saving resized images and generating AI tags...")
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
                    
                    # Save the encoded JPEG bytes to file
                    encoded_bytes = row['encoded_image']
                    with open(processed_path, 'wb') as f:
                        f.write(encoded_bytes)
                    
                    # Generate tags and caption using the resized image array directly
                    resized_array = row['resized_image']
                    tags, caption = self.generate_tags_and_caption_from_array(resized_array)
                    
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
