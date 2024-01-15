# -*-encoding:utf-8-*-

from transformers import AutoProcessor, AutoModelForVision2Seq


class KoSmosDetect:

    def __init__(self, model_path=None, device="cuda"):
        self.model_path = model_path if model_path else 'microsoft/kosmos-2-patch14-224'

        self.model = self._model()

        self.processor = self._processor()

        self.model.to(device)

    def _model(self):
        return AutoModelForVision2Seq.from_pretrained(self.model_path)

    def _processor(self):
        return AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image, sample=True):
        prompt = '<grounding>An image of' if sample else '<grounding> Describe this image in detail:'

        inputs = self.processor(prompt=prompt, images=image, return_tensors='pt')[0]

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            inputs_ids=inputs["inputs_id"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        processed_text, entities = self.processor.post_process_generation(generated_text)

        return processed_text, entities


if __name__ == "__main__":
    print("hello")
