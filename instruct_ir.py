from .instruct_ir_model import InstructIRModel
from PIL import Image
import numpy as np
import torch

def pil2tensor(image):
  return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class LoadInstructIRModel:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "device": (["cpu", "cuda", ], {"default": "cuda"}),
            }
        }


    RETURN_TYPES = ("INSTRUCTIR_MODEL",)
    FUNCTION = "get_model"
    CATEGORY = "fofo"

    def get_model(self, device):       
            
        return (InstructIRModel(device=device),)


class InstructIRProcess:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INSTRUCTIR_MODEL", ),
                "image": ("IMAGE", "IMAGE_URL"),
                "prompt": ([
                    "please I want this image for my photo album, can you edit it as a photographer",
                    "my image is too dark, can you fix it?",
                    "How can I remove the fog and mist from this photo?",
                    "enhance the colors",
                    "I need to enhance the size and quality of this image."
                    
                    ], {
                    "default": "enhance the colors",
                    "multiline": True,
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_value"
    CATEGORY = "fofo"

    def get_value(self, model, image, prompt, custom_prompt):
        # image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(),0,255).astype(np.uint8))
        if len(custom_prompt.strip())>0:
            prompt = custom_prompt
        output_image = model.process_img(tensor2pil(image[0]), prompt)
        output_image = output_image.convert("RGB")
        return (pil2tensor(output_image),)
