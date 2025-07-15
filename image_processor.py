import daft
import os
import json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import hashlib
from urllib.parse import urlparse

class ImageProcessor:
    def __init__(self):
        # Initialize the image captioning model
        self.processor = None
        self.model = None
        self.load_model()
        
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
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        
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
        
        return df.collect()
    
    def convert_uri_to_path(self, uri_path):
        """Convert file:// URI to local file path"""
        if uri_path.startswith('file://'):
            parsed = urlparse(uri_path)
            return parsed.path
        return uri_path
    
    def resize_image(self, image_path, target_size=(224, 224)):
        """Resize image to target size while maintaining aspect ratio"""
        try:
            with Image.open(image_path) as img:
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
                return new_img
                
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return None
    
    def generate_tags_and_caption(self, image):
        """Generate tags and caption for an image using BLIP model"""
        if self.model is None or self.processor is None:
            # Fallback to basic tags if model is not loaded
            return ["image"], "An image file"
        
        try:
            # Generate caption
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Generate basic tags from caption
            common_objects = [
                # People
                'person', 'people', 'man', 'woman', 'child', 'baby', 'boy', 'girl', 'adult', 'human', 'face', 'head',
                'family', 'group', 'crowd', 'individual', 'portrait', 'selfie', 'couple', 'friend', 'friends',
                
                # Animals
                'dog', 'cat', 'bird', 'animal', 'pet', 'horse', 'cow', 'sheep', 'pig', 'chicken', 'duck', 'goose',
                'fish', 'shark', 'whale', 'dolphin', 'bear', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra',
                'monkey', 'rabbit', 'squirrel', 'deer', 'fox', 'wolf', 'mouse', 'rat', 'hamster', 'guinea pig',
                'butterfly', 'bee', 'spider', 'insect', 'wildlife', 'zoo', 'farm', 'domestic', 'wild',
                
                # Vehicles
                'car', 'bike', 'bicycle', 'motorcycle', 'truck', 'bus', 'van', 'taxi', 'vehicle', 'transport',
                'train', 'airplane', 'plane', 'helicopter', 'boat', 'ship', 'yacht', 'sailboat', 'submarine',
                'scooter', 'skateboard', 'roller', 'wheel', 'tire', 'engine', 'traffic', 'road',
                
                # Buildings & Architecture
                'house', 'building', 'home', 'structure', 'architecture', 'tower', 'bridge', 'castle', 'church',
                'school', 'hospital', 'office', 'store', 'shop', 'mall', 'restaurant', 'cafe', 'hotel',
                'apartment', 'skyscraper', 'barn', 'garage', 'warehouse', 'factory', 'museum', 'library',
                'stadium', 'theater', 'cinema', 'bank', 'station', 'airport', 'construction', 'brick', 'stone',
                
                # Nature & Plants
                'tree', 'flower', 'plant', 'leaf', 'grass', 'forest', 'garden', 'branch', 'root', 'seed',
                'rose', 'tulip', 'daisy', 'sunflower', 'lily', 'orchid', 'bush', 'hedge', 'vine', 'moss',
                'fern', 'palm', 'pine', 'oak', 'maple', 'willow', 'bamboo', 'cactus', 'succulent', 'herb',
                
                # Landscape & Geography
                'beach', 'ocean', 'sea', 'water', 'mountain', 'hill', 'valley', 'lake', 'river', 'stream',
                'waterfall', 'pond', 'pool', 'desert', 'sand', 'rock', 'cliff', 'cave', 'island', 'coast',
                'shore', 'bay', 'harbor', 'field', 'meadow', 'prairie', 'plain', 'plateau', 'canyon', 'gorge',
                
                # Weather & Sky
                'sky', 'cloud', 'clouds', 'sun', 'moon', 'star', 'stars', 'sunset', 'sunrise', 'dawn', 'dusk',
                'rain', 'snow', 'storm', 'lightning', 'thunder', 'wind', 'fog', 'mist', 'rainbow', 'weather',
                
                # Food & Drink
                'food', 'meal', 'plate', 'bowl', 'cup', 'glass', 'bottle', 'fruit', 'apple', 'banana', 'orange',
                'grape', 'strawberry', 'cherry', 'peach', 'pear', 'lemon', 'lime', 'vegetable', 'tomato',
                'carrot', 'potato', 'onion', 'pepper', 'lettuce', 'cabbage', 'broccoli', 'corn', 'bread',
                'cake', 'cookie', 'pie', 'pizza', 'sandwich', 'burger', 'meat', 'chicken', 'beef', 'pork',
                'fish', 'seafood', 'cheese', 'milk', 'egg', 'pasta', 'rice', 'soup', 'salad', 'dessert',
                'coffee', 'tea', 'juice', 'wine', 'beer', 'soda', 'water', 'drink', 'beverage',
                
                # Furniture & Objects
                'table', 'chair', 'bed', 'sofa', 'couch', 'desk', 'furniture', 'seat', 'bench', 'stool',
                'cabinet', 'shelf', 'bookshelf', 'dresser', 'wardrobe', 'mirror', 'lamp', 'light', 'window',
                'door', 'wall', 'floor', 'ceiling', 'curtain', 'blind', 'carpet', 'rug', 'pillow', 'cushion',
                'blanket', 'sheet', 'towel', 'clock', 'picture', 'painting', 'frame', 'vase', 'pot', 'pan',
                
                # Rooms & Spaces
                'room', 'kitchen', 'bedroom', 'bathroom', 'living', 'dining', 'office', 'study', 'garage',
                'basement', 'attic', 'balcony', 'patio', 'deck', 'porch', 'hallway', 'corridor', 'closet',
                'pantry', 'laundry', 'indoor', 'interior', 'inside',
                
                # Outdoor & Settings
                'outdoor', 'outside', 'exterior', 'yard', 'lawn', 'driveway', 'sidewalk', 'street', 'road',
                'highway', 'path', 'trail', 'park', 'playground', 'garden', 'backyard', 'frontyard', 'fence',
                'gate', 'mailbox', 'city', 'town', 'village', 'urban', 'rural', 'suburban', 'downtown',
                'neighborhood', 'plaza', 'square', 'market', 'fair', 'festival', 'event', 'crowd', 'public',
                
                # Colors
                'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
                'gray', 'grey', 'silver', 'gold', 'beige', 'tan', 'maroon', 'navy', 'teal', 'turquoise',
                'violet', 'magenta', 'cyan', 'lime', 'olive', 'color', 'colorful', 'bright', 'dark', 'light',
                
                # Activities & Actions
                'sitting', 'standing', 'walking', 'running', 'jumping', 'playing', 'eating', 'drinking',
                'sleeping', 'reading', 'writing', 'cooking', 'cleaning', 'working', 'studying', 'talking',
                'laughing', 'smiling', 'dancing', 'singing', 'swimming', 'driving', 'riding', 'flying',
                'climbing', 'hiking', 'camping', 'fishing', 'hunting', 'shopping', 'traveling', 'vacation',
                
                # Technology & Objects
                'computer', 'laptop', 'phone', 'mobile', 'tablet', 'screen', 'monitor', 'keyboard', 'mouse',
                'camera', 'photo', 'picture', 'video', 'television', 'tv', 'radio', 'speaker', 'headphone',
                'watch', 'clock', 'calculator', 'remote', 'charger', 'cable', 'wire', 'battery', 'device',
                
                # Sports & Recreation
                'sport', 'game', 'ball', 'football', 'soccer', 'basketball', 'baseball', 'tennis', 'golf',
                'hockey', 'volleyball', 'swimming', 'running', 'cycling', 'skiing', 'surfing', 'skateboard',
                'gym', 'fitness', 'exercise', 'workout', 'team', 'player', 'athlete', 'competition', 'match',
                
                # Clothing & Accessories
                'clothing', 'clothes', 'shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'sweater',
                'hat', 'cap', 'shoes', 'boots', 'sandals', 'socks', 'gloves', 'scarf', 'belt', 'tie',
                'jewelry', 'necklace', 'ring', 'bracelet', 'earring', 'watch', 'bag', 'purse', 'backpack',
                
                # Art & Culture
                'art', 'painting', 'drawing', 'sculpture', 'statue', 'gallery', 'museum', 'culture',
                'music', 'instrument', 'guitar', 'piano', 'violin', 'drum', 'concert', 'performance',
                'book', 'magazine', 'newspaper', 'text', 'writing', 'letter', 'sign', 'poster', 'banner'
            ]
            
            tags = []
            caption_lower = caption.lower()
            for obj in common_objects:
                if obj in caption_lower:
                    tags.append(obj)
            
            # Add some basic tags if none found
            if not tags:
                tags = ['image']
            
            return tags, caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
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
        """Process all images in a folder using Daft"""
        try:
            # Update job status
            processing_jobs[job_id]['status'] = 'discovering'
            
            # Find all images using Daft
            print(f"Discovering images in {folder_path}...")
            image_files = self.find_images(folder_path)
            
            total_images = len(image_files)
            processing_jobs[job_id]['total_images'] = total_images
            processing_jobs[job_id]['status'] = 'processing'
            
            if total_images == 0:
                processing_jobs[job_id]['status'] = 'completed'
                processing_jobs[job_id]['message'] = 'No images found in the specified folder'
                return
            
            print(f"Found {total_images} images to process")
            
            # Create output directory
            output_dir = 'processed_images'
            os.makedirs(output_dir, exist_ok=True)
            
            # Process images
            processed_images = []
            
            for i, row in enumerate(image_files):
                uri_path = row['path']
                image_path = self.convert_uri_to_path(uri_path)  # Convert URI to file path
                print(f"Processing {i+1}/{total_images}: {image_path}")
                
                image_data = self.process_single_image(image_path, output_dir)
                if image_data:
                    processed_images.append(image_data)
                
                # Update progress
                processing_jobs[job_id]['processed_images'] = i + 1
                processing_jobs[job_id]['progress'] = int((i + 1) / total_images * 100)
            
            # Save results to JSON
            output_data = {
                'metadata': {
                    'processed_date': datetime.now().isoformat(),
                    'source_folder': folder_path,
                    'total_images': len(processed_images)
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
            
            print(f"Processing completed! Processed {len(processed_images)} images")
            
        except Exception as e:
            print(f"Error in process_folder: {e}")
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = str(e)
