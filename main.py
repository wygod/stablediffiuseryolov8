# -*- encoding:utf-8-*-
import os
import argparse
import operator
from omegaConf import OmegaConf

from kosmos_generate import kosmos_generate
from owlv2_generate import owlv2_generate
from stable_diffuser_self.diffuser_v2_sdxl import SDGan


def generate_text_to_image(generate_model, prompt, negative_prompt, image_number, width=512, height=512):
    images = generate_model.processor(prompt, negative_prompt, image_number, width, height)
    return images


def check_file_exist_not(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def receive_argparse():
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument('--generate_mode',
                           type=str,
                           default='stabilityai/sdxl-turbo',
                           help='please input the stable diffuser mode path')

    arg_parse.add_argument('--prompt_path',
                           type=str,
                           help='prompt text or prompt file, if prompt file, that will contain \
                           prompt and negative prompt')

    arg_parse.add_argument("--save_dir",
                           type=str,
                           default='images',
                           help='this image save path')

    arg_parse.add_argument('--W',
                           type=int,
                           default=512,
                           help='this image width')

    arg_parse.add_argument('--H',
                           type=str,
                           default=512,
                           help='this image height')

    arg_parse.add_argument('--seed',
                           type=int,
                           default=42,
                           help='random for generate image object')

    arg_parse.add_argument('--model_path',
                           type=str,
                           default='google/owlv2-base-patch16-ensemble',
                           help='detect object from image')

    arg_parse.add_argument('--use_owlv2',
                           type=bool,
                           default=False,
                           help='use owlv2 or kosmos')

    return arg_parse.parse_args()


def check_prompt(prompt):

    prompt_list = []

    if operator.contains(prompt, "/"):
        if os.path.isfile(prompt):
            with open(prompt, "r") as f_prompt:
                prompt_list.extend(f_prompt.readlines())
        else:
            raise Exception("{} is not file, please offer prompt file".format(prompt))

    else:
        prompt_list.extend(prompt.split(" "))

    return prompt_list




if __name__ == "__main__":

    data = OmegaConf.load('prompt.yaml')

    # base_path = os.path.abspath(__name__)
    #
    # arg = receive_argparse()
    #
    # save_dir = arg.save_dir if operator.contains(arg.save_dir, "/") else os.path.join(base_path, arg.save_dir)
    # check_file_exist_not(save_dir)
    #
    #
    # arg.prompt_path
    #
    #
    # prompts = []
    # if os.path.isfile(arg.class_name):
    #     with open(arg.class_name, "r") as f:
    #         prompts.extend(f.readlines())
    # else:
    #     raise Exception("prompt file no")
    #
    # assert prompts
    #
    # if arg.negative_prompt != '':
    #     pass
    #
    # sd_generate = SDGan(model_path=arg.generate_mode)
    #
    # for prompt in prompts:
    #     generate_text_to_image(sd_generate, prompt=prompt,)
