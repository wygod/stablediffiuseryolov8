#-*- encoding:utf-8-*-
import os
import torch
import argparse
import random
from diffusers import DiffusionPipeline

def text_to_image(prompts, model_file, image_path):
    #"stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_file, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()
    # prompt = "An astronaut riding a green horse"
	
    for i in range(50):
        prompt = random.sample(prompts, 1)
        images = pipe(prompt=prompt[0],num_inference_steps=4,guidance_scale=0.0).images[0]
        images.save(os.path.join(image_path, "image_{}.jpg".format(i)))
        print(i)
    #pipe.release(empty_cuda_cache=True)

def argparse_args():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate and annotate images.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="generated_dataset",
        help="Directory to save generated images and annotations",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        help="Image generator to use",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        help="List of object names for prompt generation",
    )

    return parser.parse_args()

if __name__=="__main__":
    arg = argparse_args()
    #text = ['a photo of a human on {}'.format(i) for i in arg.class_names.split(',')]
    #text = ['p photo of five human is riding in the city center.','a photo of a human is riding Harley Davidson on road.','a photo of a human is riding Harley Davidson on mountain road, creating a highway monitor picture scene.','a photo of a human is skating at the skating rink, creating a skating rink.','a photo of a waitress is serving food to a worker, creating a restaurant.','a photo of a motorcycle accident occurred on the road.','a photo of ten man are playing basketball on the basketball court.','a photo of a child is catching fish in the river.','a photo of many construction workers are cleaning up the construction site after get off work.','a photo of a man is playing indoor golf, creating a indoor golf.','a photo of many old people are dancing in the square.','a photo of a lot child is running on the park.','a photo of a motorcycle is parked on the road at sunset.','a photo of lumberjacks are cutting down trees on mountain at sunset','a photo of a couple and their children are walking on the street in the morning','a photo of a traffic policeman is punishing a motorcyclist','a photo of an Indian motorcycle travels on a kilometer next to a soybean field on the blue sky']
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir)

    text_to_image(text, arg.model_file, arg.save_dir)
	
