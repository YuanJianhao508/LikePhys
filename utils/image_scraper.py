import os
import requests
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
import random
import logging

class ImageScraper:
    def __init__(self, output_dir="data/images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def download_unsplash_images(self, query, num_images=10):
        """
        Download images from Unsplash
        Args:
            query: search term
            num_images: number of images to download
        Returns:
            list of downloaded image paths
        """
        # Note: In production, you should use the official Unsplash API
        base_url = f"https://unsplash.com/ngetty/v3/search/images/creative"
        params = {
            "fields": "id,title,thumb,url",
            "phrase": query,
            "sort": "best",
            "page": 1,
            "per_page": num_images
        }
        
        image_paths = []
        try:
            response = requests.get(base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            for idx, item in enumerate(response.json()['images']):
                image_url = item['url']
                save_path = os.path.join(self.output_dir, f"unsplash_{query}_{idx+1}.jpg")
                
                if self._download_image(image_url, save_path):
                    image_paths.append(save_path)
                
                # Be nice to the server
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error downloading from Unsplash: {str(e)}")
        
        return image_paths

    def download_pexels_images(self, query, num_images=10):
        """
        Download images from Pexels
        Args:
            query: search term
            num_images: number of images to download
        Returns:
            list of downloaded image paths
        """
        # Note: In production, you should use the official Pexels API
        base_url = f"https://api.pexels.com/v1/search"
        params = {
            "query": query,
            "per_page": num_images
        }
        
        image_paths = []
        try:
            response = requests.get(base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            for idx, photo in enumerate(response.json()['photos']):
                image_url = photo['src']['original']
                save_path = os.path.join(self.output_dir, f"pexels_{query}_{idx+1}.jpg")
                
                if self._download_image(image_url, save_path):
                    image_paths.append(save_path)
                
                # Be nice to the server
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error downloading from Pexels: {str(e)}")
        
        return image_paths

    def _download_image(self, url, save_path):
        """
        Download and save a single image
        Args:
            url: image URL
            save_path: where to save the image
        Returns:
            bool: whether download was successful
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Open the image and convert to RGB
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Resize if too large (SVD typically expects reasonable sizes)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            img.save(save_path, 'JPEG', quality=95)
            self.logger.info(f"Successfully downloaded: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return False

def download_test_images():
    """
    Download a diverse set of test images for physics analysis
    """
    scraper = ImageScraper()
    
    # Categories good for motion analysis
    categories = [
        "pendulum",
        "waterfall",
        "spinning top",
        "bouncing ball",
        "swinging",
        "falling objects",
        "rolling ball",
        "flowing river",
        "flying bird",
        "jumping animal"
    ]
    
    all_images = []
    for category in tqdm(categories, desc="Downloading images"):
        # Try both sources for each category
        images = scraper.download_unsplash_images(category, num_images=2)
        images.extend(scraper.download_pexels_images(category, num_images=2))
        all_images.extend(images)
        
        # Be extra nice to servers
        time.sleep(2)
    
    return all_images

if __name__ == "__main__":
    # Download test images
    print("Downloading test images...")
    image_paths = download_test_images()
    print(f"\nSuccessfully downloaded {len(image_paths)} images") 