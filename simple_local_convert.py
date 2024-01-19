#-*- encoding:utf-8 -*-
import argparse
import json
import os
import shutil

from PIL import Image
import xml.dom.minidom as minidom
from xml.etree import ElementTree as et


def get_w_h(img_path):
    print("----")
    print(img_path)
    return Image.open(img_path).size


def create_check_exists_file(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def create_yoyo_format(file_path, annotations_path, miss_list):
    if not file_path.endswith("json"):
        raise Exception("data is error")
    label_list = []
    with open(file_path, "r") as f:
        data = json.loads("".join(f.readlines()))

    for key, value in data.items():
        if "class_names" == key:
            label_list.extend(value)
        else:
            if not value:
                miss_list.append(key)
                continue
            image_size = get_w_h(os.path.join(file_path.rsplit("/", 1)[0], key))
            if not value["boxes"]:
                continue

            create_xml_file(key, zip([data["class_names"][i] for i in value["labels"]], value["boxes"]), image_size, annotations_path)

    with open(os.path.join(annotations_path, "class.txt"), "w") as f:
        f.write("\n".join(label_list))


def move_image(file_path, image_path, miss_list):
    file_list = os.listdir(file_path)
    for i_path in file_list:
        if i_path.endswith("jpg") and not i_path in miss_list:
            shutil.copy(os.path.join(file_path, i_path), image_path)


def create_xml_file(image_id, content, image_size, output_path):
    root = et.Element('annotation')
    folder = et.SubElement(root, 'folder')
    folder.text = 'Images'

    filename = et.SubElement(root, 'filename')
    filename.text = str(image_id)

    path = et.SubElement(root, 'path')
    path.text = '../Images/{}'.format(image_id)

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

    for meta_name, boxs in content:
        if len(boxs) > 1:
            for i_box in boxs:
                object_text = et.SubElement(root, 'object')
                name = et.SubElement(object_text, 'name')
                name.text = meta_name
                pose = et.SubElement(object_text, 'pose')
                pose.text = 'Unspecified'
                truncated = et.SubElement(object_text, 'truncated')
                truncated.text = '0'
                difficult = et.SubElement(object_text, 'difficult')
                difficult.text = '0'
                box = et.SubElement(object_text, 'bndbox')
                x_min = et.SubElement(box, 'xmin')
                x_min.text = str(int(i_box[0]))
                y_min = et.SubElement(box, 'ymin')
                y_min.text = str(int(i_box[1]))
                x_max = et.SubElement(box, 'xmax')
                x_max.text = str(int(i_box[2]))
                y_max = et.SubElement(box, 'ymax')
                y_max.text = str(int(i_box[3]))
        else:
            object_text = et.SubElement(root, 'object')
            name = et.SubElement(object_text, 'name')
            name.text = meta_name
            pose = et.SubElement(object_text, 'pose')
            pose.text = 'Unspecified'
            truncated = et.SubElement(object_text, 'truncated')
            truncated.text = '0'
            difficult = et.SubElement(object_text, 'difficult')
            difficult.text = '0'
            box = et.SubElement(object_text, 'bndbox')
            x_min = et.SubElement(box, 'xmin')
            x_min.text = str(int(boxs[0]))
            y_min = et.SubElement(box, 'ymin')
            y_min.text = str(int(boxs[1]))
            x_max = et.SubElement(box, 'xmax')
            x_max.text = str(int(boxs[2]))
            y_max = et.SubElement(box, 'ymax')
            y_max.text = str(int(boxs[3]))

    rough_str = et.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    new_str = reparsed.toprettyxml(indent='\t')

    f = open(os.path.join(output_path, image_id.replace('.jpg', '.xml')), 'w', encoding='utf-8')
    f.write(new_str)
    f.close()


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("input", type=str, help="input datadreamer path")
    arg.add_argument("output", type=str, help="output datadreamer path")

    arg_input = arg.parse_args()
    annotations_path = "{}/Annotations".format(arg_input.output)
    create_check_exists_file(annotations_path)
    print(arg_input.output)
    print(arg_input.input)
    images_path = "{}/Images".format(arg_input.output)
    create_check_exists_file(images_path)
    input_paths = os.path.join(arg_input.input, "annotations.json")
    print(input_paths)
    miss_list = []
    create_yoyo_format(input_paths, annotations_path, miss_list)
    print(miss_list)
    move_image(arg_input.input, images_path, miss_list)

