import base64
from io import BytesIO
from PIL import Image, ImageDraw
import cv2
import numpy as np
from joblib import Parallel, delayed
# from diffdvr import renderer_dtype_torch, renderer_dtype_np
# import torch.nn.functional as F
# from torchvision import transforms

# Image to base64 conversion function
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def add_red_border(img, border_size=4):
    """
    Add red border to image
    
    Args:
        img (PIL.Image): Input image
        border_size (int): Border width
        
    Returns:
        PIL.Image: Image with border added
    """
    try:
        img.load()
        # Create image copy to avoid modifying original image and file pointer issues
        img_copy = img.copy()
        
        # Draw border
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        color = (255, 0, 0)  # Red
        
        # Draw four border lines
        draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border_size)
        return img_copy
        
    except Exception as e:
        # If an error occurs, try another method
        print(f"Warning: Error adding border: {e}, trying alternative method")
        try:
            # Convert to numpy array then back, force reload image
            img_array = np.array(img)
            img_new = Image.fromarray(img_array)
            
            # Draw border on new image
            draw = ImageDraw.Draw(img_new)
            width, height = img_new.size
            color = (255, 0, 0)  # Red
            draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border_size)
            return img_new
            
        except Exception as e2:
            # If still fails, return original image and log error
            print(f"Error adding border (alternative method also failed): {e2}")
            return img
        
def __combine_image(img1, img2, middle_img=None):
    """
    Combine two images together. If middle_img is provided, place it between the two images.
    Return image data in memory.
    
    Args:
        img1: Left image
        img2: Right image
        middle_img: Optional, middle image
        
    Returns:
        str: Base64 encoding of combined image
    """
    # Ensure all images have the same height
    if middle_img is not None:
        heights = [img1.size[1], img2.size[1], middle_img.size[1]]
        new_height = min(heights)
        
        # Resize all images to the same height
        if img1.size[1] != new_height:
            img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
        if img2.size[1] != new_height:
            img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))
        if middle_img.size[1] != new_height:
            middle_img = middle_img.resize((int(middle_img.size[0] * new_height / middle_img.size[1]), new_height))
        
        # Combine three images
        total_width = img1.size[0] + middle_img.size[0] + img2.size[0]
        combined_img = Image.new("RGB", (total_width, new_height))
        combined_img.paste(middle_img, (0, 0))
        combined_img.paste(img1, (middle_img.size[0], 0))
        combined_img.paste(img2, (img1.size[0] + middle_img.size[0], 0))
    else:
        # Original logic: only combine two images
        if img1.size[1] != img2.size[1]:
            new_height = min(img1.size[1], img2.size[1])
            img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
            img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))
            
        # Combine two images
        total_width = img1.size[0] + img2.size[0]
        combined_img = Image.new("RGB", (total_width, img1.size[1]))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.size[0], 0))
    
    # Convert result to base64
    img_base64 = image_to_base64_pil(combined_img)
    return img_base64


def combine_image(img1, img2, middle_img=None, add_border=True):
    if add_border:
        bordered_img1 = add_red_border(img1)
        bordered_img2 = add_red_border(img2)
    else:
        bordered_img1 = img1
        bordered_img2 = img2
    if middle_img is not None and add_border:
        bordered_middle_img = add_red_border(middle_img)
    else:
        bordered_middle_img = middle_img.copy() if middle_img else None
    # if bordered_middle_img:
    #     print("aaaaaaaaaaaaa")
    # else:
    #     if middle_img:
    #         print("bbbbbbbbbbbbb")
    #     else:
    #         print("cccccccccccc")
    #         import traceback
    #         traceback.print_stack()
    return __combine_image(bordered_img1, bordered_img2, bordered_middle_img), __combine_image(bordered_img2, bordered_img1, bordered_middle_img)

#deprecated
def image_to_base64_nparray(img):
    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Convert numpy array to JPEG format byte stream
    _, buffer = cv2.imencode('.jpg', img_array)

    # Convert directly to Base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def image_to_base64_pil(img):
    """PIL direct conversion to Base64, avoiding OpenCV color issues"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")  # Can also use "PNG"
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def apply_semi_transparent_background(image, opacity=0.4, bg_color=(255, 255, 255)):
    """
    Process image to be semi-transparent and paste onto background with specified color
    
    Args:
        image (PIL.Image): Input image
        opacity (float): Opacity value, range 0-1, 1 means completely opaque
        bg_color (tuple): Background color, default is white (255,255,255)
    
    Returns:
        PIL.Image: Processed image
    """
    
    # Ensure image is a PIL Image object
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image object")
    
    # Create semi-transparent effect
    background_rgba = image.convert('RGBA')
    pixels = background_rgba.load()
    width, height = background_rgba.size
    
    # Modify transparency of each pixel
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (r, g, b, int(255 * opacity))
    
    # Create background with specified color
    bg_img = Image.new('RGBA', (width, height), bg_color + (255,))  # Add completely opaque alpha channel
    
    # Composite semi-transparent image onto background
    result = Image.alpha_composite(bg_img, background_rgba)
    
    # Convert to RGB format
    return result.convert('RGB')

def __create_gaussian_window(window_size, sigma=1.5):
    # Use Gaussian filter to create window
    gauss = torch.Tensor([np.exp(-0.5 * (x - window_size // 2) ** 2 / sigma ** 2) for x in range(window_size)])
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # Change to 4D shape
    return window

def concat_images(images):
    if not images:
        return
    
    # Convert all images to NumPy arrays and concatenate horizontally
    image_arrays = [np.array(img) for img in images]
    stacked_image = np.hstack(image_arrays)

    # Convert back to PIL.Image and save
    new_image = Image.fromarray(stacked_image)
    return new_image

# class SSIMCalculator:
#     def __init__(self, window_size=11, device="cuda"):
#         self.device = device
#         self.window = self.__create_gaussian_window(window_size).to(device)
#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.ToTensor()
#         ])
#         self.val_range = 255.0

#     def preprocess(self, image):
#         """Preprocess single image -> [1, 1, H, W]"""
#         tensor = self.transform(image).unsqueeze(0) * self.val_range
#         return tensor

#     def batch_compute_ssim(self, images_tensor):
#         """Batch compute SSIM for all image pairs"""
#         # images_tensor shape: [N, 1, H, W]
#         N = images_tensor.shape[0]
        
#         # Expand to [N, N, 1, H, W] for pairwise comparison
#         img1 = images_tensor.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [N, N, 1, H, W]
#         img2 = images_tensor.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, N, 1, H, W]
        
#         # Compute batch version of SSIM
#         mu1 = F.conv2d(img1.flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1)
#         mu2 = F.conv2d(img2.flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1)
        
#         sigma1_sq = F.conv2d((img1 * img1).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu1**2
#         sigma2_sq = F.conv2d((img2 * img2).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu2**2
#         sigma12 = F.conv2d((img1 * img2).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu1 * mu2
        
#         C1 = (0.01 * self.val_range) ** 2
#         C2 = (0.03 * self.val_range) ** 2
        
#         numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
#         denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
#         ssim_map = numerator / (denominator + 1e-8)  # Avoid division by zero
#         ssim_values = ssim_map.mean(dim=(1, 2, 3)).view(N, N)
        
#         # Set diagonal to zero (i == j cases)
#         mask = torch.eye(N, dtype=torch.bool, device=self.device)
#         ssim_values.masked_fill_(mask, 0.0)
        
#         return ssim_values

#     def __create_gaussian_window(self, window_size, sigma=1.5):
#         gauss = torch.Tensor([
#             np.exp(-0.5 * (x - window_size//2)**2 / sigma**2)
#             for x in range(window_size)
#         ])
#         window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
#         window = window / window.sum()
#         return window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    