# -*-encoding:utf-8-*-
import os
import shutil
import argparse

import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq


import xml.dom.minidom as minidom
from xml.etree import ElementTree as et



def create_file_not_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def move_image(input_path, output_path, del_list):
    file_list = os.listdir(input_path)
    output_path = os.path.join(output_path, 'Images')
    create_file_not_exists(output_path)

    image_list = []
    for i_path in file_list:
        if i_path.endswith(".jpg") and i_path not in del_list:
            shutil.copy(os.path.join(input_path, i_path), output_path)
            image_list.append(os.path.join(output_path, i_path.replace("image_", "")))
    return image_list


def get_label_index(meta_name, meta_label):
    label_list = [i_meta_name for i_meta_name in meta_label if i_meta_name in meta_name.split(" ")]

    if label_list:
        return meta_label.index(label_list[0]), label_list
    return None, None


def create_xml_file(image, content, image_size, output_path, label_list):

    inner_1 = time.time()

    root = et.Element('annotation')
    folder = et.SubElement(root, 'folder')
    folder.text = 'Images'

    filename = et.SubElement(root, 'filename')
    filename.text = image

    path = et.SubElement(root, 'path')
    path.text = '../Images/{}'.format(image)

    source = et.SubElement(root, 'source')
    database_sub = et.SubElement(source, 'database')
    database_sub.text = 'Unknown'

    size = et.SubElement(root, 'size')
    width = et.SubElement(size, 'width')
    width.text = str(image_size[0])
    height = et.SubElement(size, 'height')
    height.text = str(image_size[1])
    depth = et.SubElement(size, 'depth')
    depth.text = '3'

    segmented = et.SubElement(root, 'segmented')
    segmented.text = '0'

    print("innner_ 1---- time: {}".format(time.time() - inner_1))

    flag = 0

    for meta_name, size_tuple, boxs in content:
        #meta_realize_name = [i for i in label_list if i in meta_name]
        if meta_name:#meta_realize_name:
            print(len(boxs))
            for i_box in boxs:
                inner_2 = time.time()
                object_text = et.SubElement(root, 'object')
                name = et.SubElement(object_text, 'name')
                name.text = meta_name#meta_realize_name[0]
                pose = et.SubElement(object_text, 'pose')
                pose.text = 'Unspecified'
                truncated = et.SubElement(object_text, 'truncated')
                truncated.text = '0'
                difficult = et.SubElement(object_text, 'difficult')
                difficult.text = '0'
                box = et.SubElement(object_text, 'bndbox')
                x_min = et.SubElement(box, 'xmin')
                x_min.text = str(int(i_box[0] * image_size[0]))
                y_min = et.SubElement(box, 'ymin')
                y_min.text = str(int(i_box[1] * image_size[1]))
                x_max = et.SubElement(box, 'xmax')
                x_max.text = str(int(i_box[2] * image_size[0]))
                y_max = et.SubElement(box, 'ymax')
                y_max.text = str(int(i_box[3] * image_size[1]))
                print("inner_2----{}".format(time.time() - inner_2))

            flag = flag + 1
    if flag:
        inner_3 =time.time()
        rough_str = et.tostring(root, 'utf-8')

        reparsed = minidom.parseString(rough_str)

        new_str = reparsed.toprettyxml(indent='\t')
        with open(os.path.join(output_path, image.replace('.jpg', '.xml')), 'w', encoding='utf-8') as f:
            f.write(new_str)
        f.close()
        print('inner_3 ----{}'.format(time.time() - inner_3))


def create_yoyo_txt_format(image, input_path, output_path, image_path, content, meta_label, meta_size):
    temp_data = []

    label_list = []

    for meta_name, size_tuple, boxs in content:
        index, meta_label_str = get_label_index(meta_name, meta_label)
        if index:
            for i in boxs:
                temp_data.append(str(index) + " " + " ".join(map(lambda x: str(x), [(i[0]+i[2]) / 2, (i[1]+i[3]) /2, (i[2]-i[0]), (i[3]-i[1])])))
                label_list.extend(label_list)

    if temp_data:
        with open(os.path.join(output_path, image.replace("jpg", 'txt')), 'w') as f:
            f.write("\n".join(temp_data))

        shutil.copy(os.path.join(input_path, image), image_path)


    return label_list


def ko_smos_image(image, ko_smos_model, ko_smos_processor, sample=True):
    prompt_2 = "<grounding>An image of" if sample else "<grounding>Describe this image in detail:"
    inputs_2 = ko_smos_processor(text=prompt_2, images=image, return_tensors="pt").to('cuda')
    generated_ids = ko_smos_model.generate(
        pixel_values=inputs_2["pixel_values"],
        input_ids=inputs_2["input_ids"],
        attention_mask=inputs_2["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs_2["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128, )

    generated_text = ko_smos_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, entities = ko_smos_processor.post_process_generation(generated_text)
    
    del inputs_2
    
    return processed_text, entities


def arg_sys():
    arg_parses = argparse.ArgumentParser()

    arg_parses.add_argument("--image_input_dir", type=str)

    arg_parses.add_argument("--image_output_dir", type=str)

    #arg_parse.add_argument("--kosmos_model_dir")

    arg_parses.add_argument("--label", type=str)

    arg_parses.add_argument("--iter_num", type=int)
    #arg_parses.parse_args()

    return arg_parses.parse_args()


if __name__ == "__main__":
    arg_parse = arg_sys()


    model = AutoModelForVision2Seq.from_pretrained("/home/microsoft/kosmos-2-patch14-224").to('cuda')

    processor = AutoProcessor.from_pretrained("/home/microsoft/kosmos-2-patch14-224")


    output_annotation_path = os.path.join(arg_parse.image_output_dir, "Annotation")
    create_file_not_exists(output_annotation_path)
    output_image_path = os.path.join(arg_parse.image_output_dir, "Images")
    create_file_not_exists(output_image_path)

    output_annotation_xml_path = os.path.join(arg_parse.image_output_dir, "Annotationxml")
    create_file_not_exists(output_annotation_xml_path)

    meta_label = arg_parse.label.split(" ")

    with open(os.path.join(output_annotation_path, 'class.txt'), 'w') as f:
        f.write('\n'.join(meta_label))

    f.close()
    with open(os.path.join(output_annotation_xml_path, 'class.txt'), 'w') as f:
        f.write('\n'.join(meta_label))

    f.close()

    image_file = os.listdir(arg_parse.image_input_dir)

    import time

    ii = 0


    entitices_list = []

    for i_image_file in image_file:
        if i_image_file.endswith("jpg"):
            start = time.time()

            image = Image.open(os.path.join(arg_parse.image_input_dir, i_image_file))

            meta_size = image.size

            print(i_image_file)

            process_start = time.time()
            processed_text, entities = ko_smos_image(image, model, processor, sample=True)
            print('---{}--processer---time: {}'.format(ii, time.time() - process_start))

            entitices_list.append((i_image_file, entities, meta_size))

            
            #yoyo_txt_start = time.time()
            #create_yoyo_txt_format(i_image_file, arg_parse.image_input_dir, output_annotation_path, output_image_path, entities, meta_label, meta_size)
            #print("----{}--yoyo_text---time: {}".format(ii, time.time() - yoyo_txt_start))


            #xml_start = time.time()
            #create_xml_file(i_image_file, entities, meta_size, output_annotation_xml_path, meta_label)
            #print('----{}--xml---time: {}'.format(ii, time.time() - xml_start))
            
            ii = ii + 1
            
            if ii> arg_parse.iter_num:

                break
            print('-----{}--total-- time: {}'.format(ii, time.time() - start))

    ii = 1
    for jk_image, entis, sizes in entitices_list:
        
        yoyo_txt_start = time.time()
        print(jk_image)
        create_yoyo_txt_format(jk_image, arg_parse.image_input_dir, output_annotation_path, output_image_path, entis, meta_label, sizes)
        print("----{}--yoyo_text---time: {}".format(ii, time.time() - yoyo_txt_start))

        #xml_start = time.time()
        #create_xml_file(i_image_file, entities, meta_size, output_annotation_xml_path, meta_label)
        #print('----{}--xml---time: {}'.format(ii, time.time() - xml_start))

        ii = ii +1
    ii = 1
    for jkk,entities,meta_size in entitices_list:

        xml_start = time.time()
        print(jkk)
        print(entities)
        create_xml_file(jkk, entities, meta_size, output_annotation_xml_path, meta_label)
        print('----{}--xml---time: {}'.format(ii, time.time() - xml_start))
        ii = ii + 1

    #with open("init.txt", 'w') as f:
    #    f.write("\n".join(entitices_list))
    #image_list = move_image(arg_parse.image_input_dir, output_image_path, del_list)
