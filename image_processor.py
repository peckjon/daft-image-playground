import daft
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import numpy as np
import torch
from daft import udf

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
            
            # Get total count without materializing the entire dataframe
            total_images = df_images.count_rows()
            print(f"Found {total_images} image files")
            
            processing_jobs[job_id]['total_images'] = total_images
            
            if total_images == 0:
                processing_jobs[job_id].update({
                    'status': 'completed',
                    'message': 'No images found in the specified folder'
                })
                return

            processing_jobs[job_id]['status'] = 'processing'
            
            os.makedirs('processed_images', exist_ok=True)
            
            # Collect all files for processing as a list
            image_files = df_images.collect().to_pylist()

            # Process in batches to handle large datasets
            batch_size = 100  # Process 100 images at a time
            num_batches = (total_images + batch_size - 1) // batch_size
            print(f"Processing {total_images} images in {num_batches} batches of {batch_size}")
            
            # Process incrementally to avoid memory issues
            processed_images = []
            total_processed = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_images)
                batch_files = image_files[start_idx:end_idx]
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} images)")
                
                try:
                    # Create dataframe for this batch
                    batch_df = daft.from_pylist([{"path": img["path"]} for img in batch_files])
                    
                    print(f"Bulk resizing batch {batch_idx + 1} with Daft...")
                    
                    # UDF to decode and normalize images (must be 1 step, for pipelining)
                    @daft.udf(return_dtype=daft.DataType.image())
                    def decode_and_normalize_udf(file_data_series):
                        import numpy as np
                        from PIL import Image
                        import io
                        
                        results = []
                        
                        # UDFs works on the Series, so we need to iterate through each value
                        for file_data in file_data_series:
                            try:
                                # Decode the image from bytes
                                image = Image.open(io.BytesIO(file_data))
                                # Convert to RGB if needed
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                # Convert to numpy array
                                image_array = np.array(image)
                                
                                # Ensure uint8 dtype
                                if image_array.dtype != np.uint8:
                                    if np.issubdtype(image_array.dtype, np.integer):
                                        info = np.iinfo(image_array.dtype)
                                        image_array = (image_array.astype(np.float32) / info.max * 255).astype(np.uint8)
                                    else:
                                        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
                                
                                results.append(image_array)
                            except Exception as e:
                                print(f"Error in decode_and_normalize_udf: {e}")
                                results.append(None)
                        
                        return results
                    
                    # Pipeline: read -> decode+normalize -> resize -> encode
                    batch_processed = (
                        batch_df
                        .with_column("file_data", daft.col("path").url.download())
                        .with_column("image_data", decode_and_normalize_udf(daft.col("file_data")))
                        .with_column("resized_image", daft.col("image_data").image.resize(224, 224))
                        .with_column("encoded_image", daft.col("resized_image").image.encode("JPEG"))
                    ).collect()
                    
                    batch_size_actual = len(batch_processed) if batch_processed else 0
                    print(f"Batch {batch_idx + 1} processed successfully ({batch_size_actual} images)")
                    
                    # Process this batch immediately instead of accumulating
                    for i, row in enumerate(batch_processed):
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
                    
                    # Clear batch data to free memory
                    batch_processed = None
                    batch_df = None
                    
                    total_processed += batch_size_actual
                    
                    # Update progress
                    processing_jobs[job_id].update({
                        'processed_images': total_processed,
                        'progress': int(total_processed / total_images * 100),
                        'current_batch': batch_idx + 1,
                        'total_batches': num_batches
                    })
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx + 1}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
            
            print(f"Completed processing all batches. Total processed: {total_processed} images")
            
            print("Aggregating final results...")
            
            # Load existing data and aggregate new images
            os.makedirs('data', exist_ok=True)
            json_path = os.path.join('data', 'processed_images.json')
            
            existing_data = None
            existing_images = []
            existing_image_ids = set()
            
            # Load existing processed images if file exists
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                        existing_images = existing_data.get('images', [])
                        existing_image_ids = {img.get('id') for img in existing_images}
                        print(f"Found {len(existing_images)} existing processed images")
                except Exception as e:
                    print(f"Error loading existing data: {e}")
                    existing_images = []
                    existing_image_ids = set()
            
            # Filter out images that already exist (based on ID)
            new_images = []
            duplicate_count = 0
            for img in processed_images:
                if img['id'] not in existing_image_ids:
                    new_images.append(img)
                else:
                    duplicate_count += 1
            
            print(f"Found {duplicate_count} duplicate images (skipped)")
            print(f"Adding {len(new_images)} new images to library")
            
            # Combine existing and new images
            all_images = existing_images + new_images
            
            # Create aggregated output data
            output_data = {
                'metadata': {
                    'last_processed_date': datetime.now().isoformat(),
                    'last_source_folder': folder_path,
                    'total_images_in_library': len(all_images),
                    'new_images_added': len(new_images),
                    'duplicates_skipped': duplicate_count,
                    'failed_images_this_run': total_images - len(processed_images),
                    'processing_history': existing_data.get('metadata', {}).get('processing_history', []) + [{
                        'date': datetime.now().isoformat(),
                        'source_folder': folder_path,
                        'images_processed': total_images,
                        'new_images_added': len(new_images),
                        'duplicates_skipped': duplicate_count
                    }] if existing_data else [{
                        'date': datetime.now().isoformat(),
                        'source_folder': folder_path,
                        'images_processed': total_images,
                        'new_images_added': len(new_images),
                        'duplicates_skipped': duplicate_count
                    }]
                },
                'images': all_images
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Update final status
            processing_jobs[job_id].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'output_file': json_path,
                'successful_images': len(processed_images),
                'new_images_added': len(new_images),
                'total_library_size': len(all_images),
                'duplicates_skipped': duplicate_count
            })
            
            print(f"Processing completed! Successfully processed {len(processed_images)} images")
            print(f"Added {len(new_images)} new images to library (skipped {duplicate_count} duplicates)")
            print(f"Total images in library: {len(all_images)}")
            
        except Exception as e:
            print(f"Error in process_folder: {e}")
            processing_jobs[job_id].update({
                'status': 'error',
                'error': str(e)
            })
