#-*-encoding:utf-8-*-
import argparse
import torch
from diffusers import StableDiffusionPipeline,DiffusionPipeline,LCMScheduler


if __name__=="__main__":
    prompt = ['many person is standing and waiting at traffic lights','a lots of car stop on the street','a lots of authentic pesron is drinking, stopping car, coffee shop, 8k','person, woman, man, human, child, car stop on the coffee shop', 'person, store, road, motorcycle','person, woman, man, human, child, car, stroe, road, motorcycle','person, woman, man, human, child, bicycle, park','person, woman, man, human, child, on the bus','worker,person, man, human,child, street','worker,person, man, human,child, salesman,saleswoman,supermarkt','worker,person, man, human, child, baskball court','person, man, human, child, indoor baskball court','person, man, human, child, skating rink']
    negative_prompt = 'bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs'
    model_id = '/home/stabilityai/sdxl-turbo'
    #"stabilityai/stable-diffusion-xl-base-1.0"
    # lcm_lora_id = "latent-consistency/lcm-lora-sdxl"


    pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
    # pipe.load_lora_weights(lcm_lora_id)# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device="cuda", dtype=torch.float16)
    it = 0
    step = 10
    for i_word in prompt:
        for i in range(step):
            images = pipe(prompt=i_word, width=512, hight=512, num_inference_steps=4, guidance_scale=1.0, num_images_per_prompt=1, negative_prompt=negative_prompt, seed=12345678).images[0]
            images.save('/home/verify_image/image_{}.jpg'.format(i+it*step))
            print(i)
        it = it + 1
    
    pipe.to("cpu")
