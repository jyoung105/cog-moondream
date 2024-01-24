# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import os
import time
import re
from threading import Thread
from PIL import Image
# import hashlib
import torch
from huggingface_hub import snapshot_download
from models import VisionEncoder, TextModel
from transformers import TextIteratorStreamer

MODEL_NAME = "vikhyatk/moondream1"
MODEL_CACHE = "model-cache"

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

class Predictor(BasePredictor):
     
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        start = time.time()
        print("Loading pipeline...")
        
        if os.path.exists(MODEL_CACHE) is False:
            snapshot_download(MODEL_NAME, local_dir=MODEL_CACHE)      
        # model_path = snapshot_download(MODEL_NAME, local_dir=MODEL_CACHE)  
        self.vision_encoder = VisionEncoder(MODEL_CACHE).to(DEVICE, dtype=DTYPE)
        self.text_model = TextModel(MODEL_CACHE).to(DEVICE, dtype=DTYPE)
        # self.vision_encoder = VisionEncoder(model_path).to(DEVICE, dtype=DTYPE)
        # self.text_model = TextModel(model_path).to(DEVICE, dtype=DTYPE)

        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt"),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=True,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model for commercial purpose."
            )
        
        # streamer = TextIteratorStreamer(self.text_model.tokenizer, skip_special_tokens=True)    
        img = Image.open(image)
        image_embeds = self.vision_encoder(img).to(DEVICE, dtype=DTYPE)
        # generation_kwargs = dict(
        #     image_embeds = image_embeds, question=prompt, streamer=streamer
        # )
        # thread = Thread(target=self.text_model.answer_question, kwargs=generation_kwargs)
        # thread.start()
        
        # buffer = ""
        # for new_text in streamer:
        #     buffer += new_text
        #     if len(buffer) > 1:
        #         output = re.sub("<$", "", re.sub("END$", "", buffer))          
        # output.strip()
        output = self.text_model.answer_question(image_embeds, prompt).strip()
        
        return output
    
    # def cached_vision_encoder(image):
    #     # Calculate checksum of the image
    #     image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    #     # Check if `image_encoder_cache/{image_hash}.pt` exists, if so load and return it.
    #     # Otherwise, save the encoded image to `image_encoder_cache/{image_hash}.pt` and return it.
    #     cache_path = f"image_encoder_cache/{image_hash}.pt"
    #     if os.path.exists(cache_path):
    #         return torch.load(cache_path).to("cuda")
    #     else:
    #         image_vec = vision_encoder(image).to("cpu", dtype=torch.float16)
    #         os.makedirs("image_encoder_cache", exist_ok=True)
    #         torch.save(image_vec, cache_path)
    #         return image_vec.to("cuda")