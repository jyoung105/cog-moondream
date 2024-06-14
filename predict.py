# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import os
import sys

import time
import numpy as np
from PIL import Image

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Deal with PIL.Image.DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# GPU global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32

# model global variables
MODEL_NAME = "vikhyatk/moondream2"
REVISION = "2024-05-20"
MODEL_CACHE = "model-cache"

class Predictor(BasePredictor):
     
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        start = time.time()
        print("[~] Loading pipeline...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            revision=REVISION
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            revision=REVISION,
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2"
        ).to(DEVICE)
        self.model.eval()
        
        self.classifier = pipeline(
            task="zero-shot-image-classification", 
            model="google/siglip-so400m-patch14-384",
            device=0
        )

        print("Setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="Describe this image in one sentence. ",
        ),
        candidate_labels: str = Input(
            description="Candidate labels separated by commas",
            default="photo, illustration, vector graphic, 3d render"
        ),
    ) -> str:
        """Run a single prediction on the model"""
        start1 = time.time()
        
        if prompt is None or image is None:
            msg = "No input, Save money"
            return msg
        
        img = Image.open(image)
        labels = candidate_labels.split(",")
        classification = self.classifier(img, candidate_labels=labels)
        output1 = labels[np.argmax(np.transpose(np.array([list(idx.values()) for idx in classification]))[0].astype(np.float64))]
        
        enc_image = self.model.encode_image(img)
        output2 = self.model.answer_question(enc_image, "Describe this image.", self.tokenizer)
        
        output = ", ".join([output1, output2])
        
        print("Finish generation in " + str(time.time()-start1) + " secs.")
        
        return output