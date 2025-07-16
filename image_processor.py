import daft
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import numpy as np
import torch

class ImageProcessor:
    def __init__(self):
        self.processor = None
        self.model = None
        self.stopwords = set()
        self.load_stopwords()
        self.load_model()
        
    def load_stopwords(self):
        """Load English stopwords from local file"""
        try:
            with open('stopwords-en.txt', 'r') as f:
                self.stopwords = {word for word in f.read().splitlines() if word}
            print(f"Loaded {len(self.stopwords)} stopwords")
        except Exception as e:
            print(f"Failed to load stopwords: {e}")
            self.stopwords = set()
        
    def load_model(self):
        """Load the BLIP model for image captioning"""
        try:
            print("Loading BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing without AI model")
    
    def find_images(self, folder_path):
        """Use Daft to find all images in folder and subfolders"""
        try:
            df = daft.from_glob_path(f"{folder_path}/**/*")
            
            # Case-insensitive extension matching
            return df.where(
                df["path"].str.lower().str.endswith(".jpg") |
                df["path"].str.lower().str.endswith(".jpeg") |
                df["path"].str.lower().str.endswith(".png") |
                df["path"].str.lower().str.endswith(".gif") |
                df["path"].str.lower().str.endswith(".bmp") |
                df["path"].str.lower().str.endswith(".webp") |
                df["path"].str.lower().str.endswith(".tiff")
            )
        except Exception as e:
            print(f"Error in find_images: {e}")
            # Return empty dataframe if glob fails
            return daft.from_pylist([])
    
    def convert_uri_to_path(self, uri_path):
        """Convert file:// URI to local file path"""
        if uri_path.startswith('file://'):
            return urlparse(uri_path).path
        return uri_path
    
    def clean_words_to_tags(self, words):
        """Convert caption words to clean tags"""
        tags = []
        for word in words:
            clean_word = ''.join(char for char in word if char.isalnum())
            if (3 <= len(clean_word) <= 20 and 
                clean_word not in self.stopwords and
                clean_word.isalpha()):
                tags.append(clean_word)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(tags)) or ['image']
    
    def generate_tags_and_caption_from_array(self, image_array):
        """Generate tags and caption for a numpy image array using BLIP model"""
        if self.model is None or self.processor is None:
            return ["image"], "An image file"
        
        try:
            if not isinstance(image_array, np.ndarray):
                print(f"Expected numpy array, got {type(image_array)}")
                return ["image"], "An image file"
            
            # Convert and normalize image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Make a writable copy of the array before converting to tensor
            image_tensor = torch.from_numpy(np.copy(image_array)).float() / 255.0
            
            # Rearrange dimensions and add batch dimension
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Generate caption
            out = self.model.generate(pixel_values=image_tensor, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            tags = self.clean_words_to_tags(caption.lower().split())
            return tags, caption
        
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ["image"], "An image file"
    
    def process_folder(self, folder_path, job_id, processing_jobs):
        """Process all images in a folder using Daft for bulk operations"""
        try:
            processing_jobs[job_id]['status'] = 'discovering'
            
            print(f"Discovering images in {folder_path}...")
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                processing_jobs[job_id].update({
                    'status': 'error',
                    'error': f'Folder does not exist: {folder_path}'
                })
                return
            
            if not os.path.isdir(folder_path):
                processing_jobs[job_id].update({
                    'status': 'error',
                    'error': f'Path is not a directory: {folder_path}'
                })
                return
            
            df_images = self.find_images(folder_path)
            
            try:
                image_files = df_images.collect()
                total_images = len(image_files)
                print(f"Found {total_images} image files")
                
                # Debug: show first few files found
                if total_images > 0:
                    print("Sample files found:")
                    for i, img in enumerate(image_files[:3]):
                        print(f"  {i+1}. {img['path']}")
                    if total_images > 3:
                        print(f"  ... and {total_images-3} more")
                else:
                    print("No image files found. Checking directory contents...")
                    # Manual check for debugging
                    all_files = []
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')):
                                all_files.append(os.path.join(root, file))
                    print(f"Manual scan found {len(all_files)} image files")
                    if all_files:
                        print("Sample manually found files:")
                        for f in all_files[:3]:
                            print(f"  {f}")
                            
            except Exception as e:
                print(f"Error collecting image files: {e}")
                processing_jobs[job_id].update({
                    'status': 'error',
                    'error': f'Failed to collect image files: {str(e)}'
                })
                return
                
            processing_jobs[job_id]['total_images'] = total_images
            
            if total_images == 0:
                processing_jobs[job_id].update({
                    'status': 'completed',
                    'message': 'No images found in the specified folder'
                })
                return
            
            print(f"Found {total_images} images to process")
            processing_jobs[job_id]['status'] = 'processing'
            
            os.makedirs('processed_images', exist_ok=True)
            
            print("Bulk resizing images with Daft...")
            
            # Pipeline: read -> decode -> resize -> encode
            df_images = (df_images
                .with_column("file_data", daft.col("path").url.download())
                .with_column("image_data", daft.col("file_data").image.decode())
                .with_column("resized_image", daft.col("image_data").image.resize(224, 224))
                .with_column("encoded_image", daft.col("resized_image").image.encode("JPEG"))
            )
            
            resized_results = df_images.collect()
            
            print("Saving images and generating AI tags...")
            processed_images = []
            
            for i, row in enumerate(resized_results):
                try:
                    local_path = self.convert_uri_to_path(row['path'])
                    filename = os.path.basename(local_path)
                    name, _ = os.path.splitext(filename)
                    image_hash = hashlib.md5(local_path.encode()).hexdigest()[:8]
                    processed_filename = f"{name}_{image_hash}.jpg"
                    processed_path = os.path.join('processed_images', processed_filename)
                    
                    # Save encoded image
                    with open(processed_path, 'wb') as f:
                        f.write(row['encoded_image'])
                    
                    # Generate tags and caption
                    tags, caption = self.generate_tags_and_caption_from_array(row['resized_image'])
                    
                    file_stats = os.stat(local_path)
                    processed_images.append({
                        'id': image_hash,
                        'filename': filename,
                        'original_path': local_path,
                        'processed_path': processed_filename,
                        'file_size': file_stats.st_size,
                        'created_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'tags': tags,
                        'caption': caption,
                        'processed_date': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Skipping failed image: {row.get('path', 'unknown')} - {e}")
                
                processing_jobs[job_id].update({
                    'processed_images': i + 1,
                    'progress': int((i + 1) / total_images * 100)
                })
            
            # Save results
            output_data = {
                'metadata': {
                    'processed_date': datetime.now().isoformat(),
                    'source_folder': folder_path,
                    'total_images': len(processed_images),
                    'failed_images': total_images - len(processed_images)
                },
                'images': processed_images
            }
            
            os.makedirs('data', exist_ok=True)
            json_path = os.path.join('data', 'processed_images.json')
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Update final status
            processing_jobs[job_id].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'output_file': json_path,
                'successful_images': len(processed_images)
            })
            
            print(f"Processing completed! Successfully processed {len(processed_images)} images")
            
        except Exception as e:
            print(f"Error in process_folder: {e}")
            processing_jobs[job_id].update({
                'status': 'error',
                'error': str(e)
            })
