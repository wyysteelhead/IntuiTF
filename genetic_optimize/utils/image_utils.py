import base64
from io import BytesIO
from PIL import Image, ImageDraw
import cv2
import numpy as np
from joblib import Parallel, delayed
# from diffdvr import renderer_dtype_torch, renderer_dtype_np
# import torch.nn.functional as F
# from torchvision import transforms

#图片转base64函数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def add_red_border(img, border_size=4):
    """
    给图片添加红色边框
    
    Args:
        img (PIL.Image): 输入图像
        border_size (int): 边框宽度
        
    Returns:
        PIL.Image: 添加边框后的图像
    """
    try:
        img.load()
        # 创建图像副本以避免修改原始图像和文件指针问题
        img_copy = img.copy()
        
        # 绘制边框
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        color = (255, 0, 0)  # 红色
        
        # 画四条边框线
        draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border_size)
        return img_copy
        
    except Exception as e:
        # 如果发生错误，尝试另一种方法
        print(f"Warning: Error adding border: {e}, trying alternative method")
        try:
            # 转换为numpy数组再转回，强制重新加载图像
            img_array = np.array(img)
            img_new = Image.fromarray(img_array)
            
            # 在新图像上绘制边框
            draw = ImageDraw.Draw(img_new)
            width, height = img_new.size
            color = (255, 0, 0)  # 红色
            draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border_size)
            return img_new
            
        except Exception as e2:
            # 如果仍然失败，返回原始图像并记录错误
            print(f"Error adding border (alternative method also failed): {e2}")
            return img
        
def __combine_image(img1, img2, middle_img=None):
    """
    将两张图片拼接到一起，如果提供了middle_img，则将其放在两张图片中间。
    返回内存中的图像数据。
    
    Args:
        img1: 左侧图片
        img2: 右侧图片
        middle_img: 可选，中间图片
        
    Returns:
        str: 拼接后图片的base64编码
    """
    # 确保所有图片的高度一致
    if middle_img is not None:
        heights = [img1.size[1], img2.size[1], middle_img.size[1]]
        new_height = min(heights)
        
        # 调整所有图片至相同高度
        if img1.size[1] != new_height:
            img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
        if img2.size[1] != new_height:
            img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))
        if middle_img.size[1] != new_height:
            middle_img = middle_img.resize((int(middle_img.size[0] * new_height / middle_img.size[1]), new_height))
        
        # 拼接三张图片
        total_width = img1.size[0] + middle_img.size[0] + img2.size[0]
        combined_img = Image.new("RGB", (total_width, new_height))
        combined_img.paste(middle_img, (0, 0))
        combined_img.paste(img1, (middle_img.size[0], 0))
        combined_img.paste(img2, (img1.size[0] + middle_img.size[0], 0))
    else:
        # 原本的逻辑：只拼接两张图片
        if img1.size[1] != img2.size[1]:
            new_height = min(img1.size[1], img2.size[1])
            img1 = img1.resize((int(img1.size[0] * new_height / img1.size[1]), new_height))
            img2 = img2.resize((int(img2.size[0] * new_height / img2.size[1]), new_height))
            
        # 拼接两张图片
        total_width = img1.size[0] + img2.size[0]
        combined_img = Image.new("RGB", (total_width, img1.size[1]))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.size[0], 0))
    
    # 将结果转换为base64
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
    # 转换 PIL Image 为 numpy 数组
    img_array = np.array(img)

    # 将 numpy 数组转换为 JPEG 格式的字节流
    _, buffer = cv2.imencode('.jpg', img_array)

    # 直接转换为 Base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def image_to_base64_pil(img):
    """PIL 直接转换 Base64，避免 OpenCV 颜色问题"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")  # 也可以用 "PNG"
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def apply_semi_transparent_background(image, opacity=0.4, bg_color=(255, 255, 255)):
    """
    将图像处理为半透明并粘贴到指定颜色的背景上
    
    Args:
        image (PIL.Image): 输入的图像
        opacity (float): 不透明度值，范围0-1，1表示完全不透明
        bg_color (tuple): 背景颜色，默认为白色(255,255,255)
    
    Returns:
        PIL.Image: 处理后的图像
    """
    
    # 确保图像为PIL图像对象
    if not isinstance(image, Image.Image):
        raise TypeError("输入必须是PIL Image对象")
    
    # 创建半透明效果
    background_rgba = image.convert('RGBA')
    pixels = background_rgba.load()
    width, height = background_rgba.size
    
    # 修改每个像素的透明度
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (r, g, b, int(255 * opacity))
    
    # 创建指定颜色的背景
    bg_img = Image.new('RGBA', (width, height), bg_color + (255,))  # 添加完全不透明的alpha通道
    
    # 将半透明图像合成到背景上
    result = Image.alpha_composite(bg_img, background_rgba)
    
    # 转换为RGB格式
    return result.convert('RGB')

def __create_gaussian_window(window_size, sigma=1.5):
    # 使用高斯滤波器创建窗口
    gauss = torch.Tensor([np.exp(-0.5 * (x - window_size // 2) ** 2 / sigma ** 2) for x in range(window_size)])
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # 变成 4D 形状
    return window

def concat_images(images):
    if not images:
        return
    
    # 将所有图片转换为 NumPy 数组并水平拼接
    image_arrays = [np.array(img) for img in images]
    stacked_image = np.hstack(image_arrays)

    # 转换回 PIL.Image 并保存
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
#         """预处理单张图像 -> [1, 1, H, W]"""
#         tensor = self.transform(image).unsqueeze(0) * self.val_range
#         return tensor

#     def batch_compute_ssim(self, images_tensor):
#         """批量计算所有图像对的 SSIM"""
#         # images_tensor 形状: [N, 1, H, W]
#         N = images_tensor.shape[0]
        
#         # 扩展为 [N, N, 1, H, W] 以便两两比较
#         img1 = images_tensor.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [N, N, 1, H, W]
#         img2 = images_tensor.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, N, 1, H, W]
        
#         # 计算 SSIM 的批量版本
#         mu1 = F.conv2d(img1.flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1)
#         mu2 = F.conv2d(img2.flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1)
        
#         sigma1_sq = F.conv2d((img1 * img1).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu1**2
#         sigma2_sq = F.conv2d((img2 * img2).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu2**2
#         sigma12 = F.conv2d((img1 * img2).flatten(0, 1), self.window, padding=self.window.size(-1)//2, groups=1) - mu1 * mu2
        
#         C1 = (0.01 * self.val_range) ** 2
#         C2 = (0.03 * self.val_range) ** 2
        
#         numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
#         denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
#         ssim_map = numerator / (denominator + 1e-8)  # 避免除以零
#         ssim_values = ssim_map.mean(dim=(1, 2, 3)).view(N, N)
        
#         # 将对角线置零（i == j 的情况）
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
    