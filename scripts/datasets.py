from sklearn.model_selection import train_test_split
from datasets import load_dataset
from PIL import Image
import io
import random


def office_home(domain):
    # Load the Office-Home dataset
    ds = load_dataset("flwrlabs/office-home")
    
    # Map domain to the corresponding subset
    domain_map = {
        'real': ds['train'].filter(lambda x: x['domain'] == 'Real World'),
        'art': ds['train'].filter(lambda x: x['domain'] == 'Art'),
        'clipart': ds['train'].filter(lambda x: x['domain'] == 'Clipart'),
        'product': ds['train'].filter(lambda x: x['domain'] == 'Product')
    }
    
    # Retrieve the class names
    class_names = ds['train'].features['label'].names
    
    # Return the filtered data for the specified domain
    if domain in domain_map:
        return class_names, domain_map[domain]
    else:
        raise ValueError(f"Invalid domain specified: {domain}. Valid options are: {list(domain_map.keys())}")
    


def convert_bytes_to_images(dataset):
    """
    Converts a dataset of byte-encoded images into a list of PIL Image objects.

    Parameters:
    - dataset: List of dictionaries, each containing 'bytes' (image data) and 'path' (file name).

    Returns:
    - List of PIL Image objects.
    """
    images = []
    for image_dict in dataset:
        try:
            # Extract image bytes
            image_bytes = io.BytesIO(image_dict['bytes'])
            
            # Open the image using PIL
            image = Image.open(image_bytes)
            
            # Append the PIL Image object to the list
            images.append(image)
        except Exception as e:
            print(f"Failed to process image {image_dict.get('path', 'unknown')}: {e}")
            images.append(None)  # Append None for failed images

    return images

