# -*-encoding:utf-8-*-

import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection


class Owlv2Detect:

    def __init__(self, model_path=None, device='cuda'):

        self.model_path =model_path if model_path else 'google/owlv2-base-patch16-ensemble'

        self.model = self._model()

        self.processor = self._processor()

        self.model.to("cuda")

    def _model(self):

        return Owlv2ForObjectDetection.from_pretrained(self.model_path)

    def _processor(self):
        return Owlv2ForObjectDetection.from_pretrained(self.model_path)

    def generate(self, prompt, image):
        inputs = self.processor(text=prompt, images=image, return_tensors=True)
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detect(outputs=outputs)
        # results [{'boxs':'', 'score':'', 'labels':''}]

        return results



