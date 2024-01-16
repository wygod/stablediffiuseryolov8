# -*-encoding:utf-8-*-

import torch
from diffusers import AutoPipelineForText2Image


class SDGan:
    def __init__(self, model_path=None, device='cuda'):
        self.model_path = model_path if model_path else 'stabilityai/sdxl-turbo'
        self.model = self._model()
        self.model.to("cuda")

    def _model(self):
        return AutoPipelineForText2Image.from_pretrained(self.model_path)

    def processor(self, prompt, negative_prompt, width=512, height=512, image_number=1):
        images = self.model(prompt=prompt,
                            num_inference_steps=4,
                            image_num_per_prompt=image_number,
                            guidance_scale=0.0,
                            width=width,
                            height=height,
                            negative_prompt=negative_prompt,
                            seed=-1,
                            ).images
        return images
