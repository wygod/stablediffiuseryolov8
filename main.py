# -*- encoding:utf-8-*-
import os
import argparse
import operator
from PIL import Image
from lxml import etree
from xml.dom import minidom
from omegaconf import OmegaConf


from kosmos_generate import kosmos_generate
from owlv2_generate import owlv2_generate
from stable_diffuser_self.diffuser_v2_sdxl import SDGan


def generate_text_to_image(generate_model, prompt, negative_prompt, image_number, seed,width=512, height=512):
    images = generate_model.processor(prompt, negative_prompt, image_number, seed, width, height)
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

    arg_parse.add_argument("--batch_image_number",
                           type=int,
                           default=1,
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

    arg_parse.add_argument('--sample',
                           type=bool,
                           default=True,
                           help='detect kosmos, must be choice')

    return arg_parse.parse_args()


def check_prompt(prompt):
    if os.path.isfile(prompt):
        prompt_content = OmegaConf.load(prompt)
    else:
        raise Exception("prompt must be yaml file")

    return prompt_content


def create_xml_file(image_id, content, image_size, output_path):
    root = etree.Element('annotation')
    folder = etree.SubElement(root, 'folder')
    folder.text = 'Images'

    filename = etree.SubElement(root, 'filename')
    filename.text = image_id

    path = etree.SubElement(root, 'path')
    path.text = '../Images/{}'.format(image_id)

    source = etree.SubElement(root, 'source')
    database_sub = etree.SubElement(source, 'database')
    database_sub.text = 'Unknown'

    size = etree.SubElement(root, 'size')
    width = etree.SubElement(size, 'width')
    width.text = str(image_size[0])
    height = etree.SubElement(size, 'height')
    height.text = str(image_size[1])
    depth = etree.SubElement(size, 'depth')
    depth.text = '3'

    segmented = etree.SubElement(root, 'segmented')
    segmented.text = '0'

    for meta_name, size_tuple, boxs in content:
        for i_box in boxs:
            object_text = etree.SubElement(root, 'object')
            name = etree.SubElement(object_text, 'name')
            name.text = meta_name
            pose = etree.SubElement(object_text, 'pose')
            pose.text = 'Unspecified'
            truncated = etree.SubElement(object_text, 'truncated')
            truncated.text = '0'
            difficult = etree.SubElement(object_text, 'difficult')
            difficult.text = '0'
            box = etree.SubElement(object_text, 'bndbox')
            x_min = etree.SubElement(box, 'xmin')
            x_min.text = str(int(i_box[0] * image_size[0]))
            y_min = etree.SubElement(box, 'ymin')
            y_min.text = str(int(i_box[1] * image_size[1]))
            x_max = etree.SubElement(box, 'xmax')
            x_max.text = str(int(i_box[2] * image_size[0]))
            y_max = etree.SubElement(box, 'ymax')
            y_max.text = str(int(i_box[3] * image_size[1]))

    rough_str = etree.tostring(root, 'utf-8')
    parse_txt = minidom.parseString(rough_str)
    new_str = parse_txt.toprettyxml(indent='\t')

    f = open(os.path.join(output_path, image_id.replace('.jpg', '.xml')), 'w', encoding='utf-8')
    f.write(new_str)
    f.close()


def create_yoyo_txt_format(image, input_path, output_path, image_path, content):
    temp_data = []

    label_list = []

    for meta_name, size_tuple, boxs in content:
        if index:
            for i in boxs:
                temp_data.append(str(index) + " " + " ".join(map(lambda x: str(x), [(i[0]+i[2]) / 2, (i[1]+i[3]) /2, (i[2]-i[0]), (i[3]-i[1])])))
                label_list.extend(label_list)

    if temp_data:
        with open(os.path.join(output_path, image.replace("jpg", 'txt')), 'w') as f:
            f.write("\n".join(temp_data))

        shutil.copy(os.path.join(input_path, image), image_path)


    return label_list

def choice_gan_mode(model_path):
    sd_gan = SDGan(model_path)
    return sd_gan


def detect_object_by_ko_smos():
    text, entities = detect_object.generate(image=temp_image, sample=arg.sample)



    pass


def detect_object_by_owlv2():
    text, entities = detect_object.generate(prompt=key_word, image=temp_image)
    pass


def choice_detect_mode(mode_path):

    if operator.contains(mode_path, 'kosmos'):
        return kosmos_generate.KoSmosDetect(model_path=mode_path)
    else:
        return owlv2_generate.Owlv2Detect(model_path=mode_path)




if __name__ == "__main__":

    base_path = os.path.abspath(__name__)

    arg = receive_argparse()

    save_dir = arg.save_dir if operator.contains(arg.save_dir, "/") else os.path.join(base_path, arg.save_dir)
    check_file_exist_not(save_dir)

    sd_model = choice_gan_mode(arg.generate_mode)

    argparse_prompt = check_prompt(arg.prompt_path)
    negative_prompt = argparse_prompt["negative_prompt"]
    step = arg.batch_number_images

    save_images_path = os.path.join(save_dir, "images")
    save_key_path = os.path.join(save_dir, "prompt")
    save_detect_path = os.path.join(save_dir, "detect")

    key_word_image = {}
    for i, key in enumerate(argparse_prompt["prompt"]):
        key_word_content = []
        for per_image in range(step):
            images = generate_text_to_image(generate_model=sd_model, prompt=argparse_prompt['prompt'][key],
                                   negative_prompt=negative_prompt, image_number=arg.image_number,
                                   seed=arg.seed, width=512, height=512)

            for per_images in range(arg.image_number):
                image_file_name = os.path.join(
                    save_images_path, 'image_{}.jpg'.format(i * step + (i + per_image) * arg.image_number + per_images))
                images[per_images].save(image_file_name)

                key_word_content.append(image_file_name)

        key_word_image[key] = key_word_content

    detect_object = choice_detect_mode(arg.mode_path)
    use_owlv2 = True if operator.contains(arg.generate_mode, 'owlv2') else False
    for key_word, per_key_word_content in key_word_image.items():
        for per_image in per_key_word_content:
            temp_image = Image.open(per_image)
            if use_owlv2:

                detect_object_by_ko_smos
            else:




    #
    # if arg.negative_prompt != '':
    #     pass
    #
    # sd_generate = SDGan(model_path=arg.generate_mode)
    #
    # for prompt in prompts:
    #     generate_text_to_image(sd_generate, prompt=prompt,)
