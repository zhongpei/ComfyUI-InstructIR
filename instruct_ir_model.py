import argparse


from PIL import Image
import os
import torch
import numpy as np
import yaml
from huggingface_hub import hf_hub_download
#from gradio_imageslider import ImageSlider

## local code
from .models.instructir import create_model
from .text.models import LanguageModel, LMHead
from .install import get_ext_dir

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class InstructIRModel(object):
    def __init__(self,device="cuda"):
        self.model , self.language_model, self.lm_head, self.device = self.load_model(device=device)

    @classmethod
    def load_model(cls,device="cuda"):
        local_dir = get_ext_dir("model", mkdir=True)
        CONFIG     = os.path.join(get_ext_dir(),"configs/eval5d.yml")
        LM_MODEL   = os.path.join(local_dir,"lm_instructir-7d.pt")
        MODEL_NAME = os.path.join(local_dir,"im_instructir-7d.pt")
        if not os.path.exists(MODEL_NAME):
            hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt", local_dir=local_dir)
        if not os.path.exists(LM_MODEL):
            hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt", local_dir=local_dir)



        # parse config file
        with open(os.path.join(CONFIG), "r") as f:
            config = yaml.safe_load(f)

        cfg = dict2namespace(config)
        if device == "cuda":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                                    middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
        model = model.to(device)
        print ("IMAGE MODEL CKPT:", MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_NAME, map_location="cpu"), strict=True)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        LMODEL = cfg.llm.model
        language_model = LanguageModel(model=LMODEL)
        lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
        lm_head = lm_head.to(device)

        print("LMHEAD MODEL CKPT:", LM_MODEL)
        lm_head.load_state_dict(torch.load(LM_MODEL, map_location="cpu"), strict=True)
        return model, language_model, lm_head, device

    def load_img (filename, norm=True,):
        img = np.array(Image.open(filename).convert("RGB"))
        if norm:
            img = img / 255.
            img = img.astype(np.float32)
        return img


    def process_img (self,image, prompt):
        img = np.array(image)
        img = img / 255.
        img = img.astype(np.float32)
        y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(self.device)

        lm_embd = self.language_model(prompt)
        lm_embd = lm_embd.to(self.device)

        with torch.no_grad():
            text_embd, deg_pred = self.lm_head (lm_embd)
            x_hat = self.model(y, text_embd)

        restored_img = x_hat.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
        restored_img = np.clip(restored_img, 0. , 1.)

        
        restored_img = (restored_img * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        return Image.fromarray(restored_img) #(image, Image.fromarray(restored_img))

