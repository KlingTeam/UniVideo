from torchvision import transforms
import numpy as np
import torch
from PIL import Image

def pad_image_pil_to_square(image_prompt):
        image_prompt = np.array(image_prompt, copy=True)
        image_prompt = torch.from_numpy(image_prompt).permute(2, 0, 1).contiguous()
        height, width = image_prompt.size(1), image_prompt.size(2)
        if height == width:
            pass
        elif height < width:
            diff = width - height
            top_pad = diff // 2
            down_pad = diff - top_pad
            left_pad = 0
            right_pad = 0
            padding_size = [left_pad, top_pad, right_pad, down_pad]
            image_prompt = transforms.functional.pad(image_prompt, padding=padding_size, fill = 255)
        else:
            diff = height - width
            left_pad = diff // 2
            right_pad = diff - left_pad
            top_pad = 0
            down_pad = 0
            padding_size = [left_pad, top_pad, right_pad, down_pad]
            image_prompt = transforms.functional.pad(image_prompt, padding=padding_size, fill = 255)
        image_prompt_pil = Image.fromarray(image_prompt.permute(1, 2, 0).numpy())
        return image_prompt_pil