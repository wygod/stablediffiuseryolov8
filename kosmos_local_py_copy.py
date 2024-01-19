# -*-encoding:utf-8-*-
import json
import os
import shutil

import torch
import argparse
from PIL import Image
from tqdm import tqdm
import xml.dom.minidom as minidom
from xml.etree import ElementTree as et
from transformers import AutoProcessor, AutoModelForVision2Seq


class LocalKOwl:
    def __init__(self, device='cuda', detail=False):
        self.prompt ='<grounding>Describe this image in detail:' if detail else "<grounding>An image of"
        self.model = self._model()
        self.processor = self._processor()
        self.device = device
        self.model.to(self.device)

    def _model(self):
        return AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")

    def _processor(self):
        return AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    def annotate(
            self, image
    ):
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        processed_text, entities = self.processor.post_process_generation(generated_text)

        return processed_text, entities

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


def move_image(input_path, output_path):
    file_list = os.listdir(input_path)
    output_path = os.path.join(output_path, 'Images')
    create_check_exists_file(output_path)

    image_list = []
    for i_path in file_list:
        if i_path.endswith(".jpg"):
            print(i_path)
            shutil.copy(os.path.join(input_path, i_path), output_path)

            image_list.append(os.path.join(output_path, i_path.replace("image_","")))
    return image_list


def create_check_exists_file(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def create_yoyo_format(image_id, content, image_size, output_path, label_list):

    temp_data = []
    
    for meta_name, size_tuple, boxs in content:
        for i in boxs:
            temp_data.append(" ".join(map(lambda x:str(x), i)))
            label_list.append(meta_name)

    with open(os.path.join(output_path, image_id.rsplit('_', 1)[-1].replace("jpg", 'txt')), 'w') as f:
        f.write("\n".join(temp_data))


def create_xml_file(image_id, content, image_size, output_path, label_list):#, meta_label):
    root = et.Element('annotation')
    folder = et.SubElement(root, 'folder')
    folder.text = 'Images'

    filename = et.SubElement(root, 'filename')
    filename.text = image_id

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

    for meta_name, size_tuple, boxs in content:
        meta_realize_name = meta_name#"".join([i_label for i_label in meta_label]) #if i_label in meta_name])
        #if meta_realize_name =="":
        #    continue

        label_list.append(meta_realize_name)

        if len(boxs)>1:
            for i_box in boxs:
                object_text = et.SubElement(root, 'object')
                name = et.SubElement(object_text, 'name')
                name.text = meta_realize_name
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
        else:
            object_text = et.SubElement(root, 'object')
            name = et.SubElement(object_text, 'name')
            name.text = meta_realize_name
            pose = et.SubElement(object_text, 'pose')
            pose.text = 'Unspecified'
            truncated = et.SubElement(object_text, 'truncated')
            truncated.text = '0'
            difficult = et.SubElement(object_text, 'difficult')
            difficult.text = '0'
            box = et.SubElement(object_text, 'bndbox')
            x_min = et.SubElement(box, 'xmin')
            x_min.text = str(int(boxs[0][0] * image_size[0]))
            y_min = et.SubElement(box, 'ymin')
            y_min.text = str(int(boxs[0][1] * image_size[1]))
            x_max = et.SubElement(box, 'xmax')
            x_max.text = str(int(boxs[0][2] * image_size[0]))
            y_max = et.SubElement(box, 'ymax')
            y_max.text = str(int(boxs[0][3] * image_size[1]))

    rough_str = et.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    new_str = reparsed.toprettyxml(indent='\t')

    f = open(os.path.join(output_path, image_id.replace('.jpg', '.xml')), 'w', encoding='utf-8')
    f.write(new_str)
    f.close()


if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--generate_save_dir", type=str, default='', help='generate image path')

    arg.add_argument("--generate_output_dir", type=str, default='', help='output image path')

    sys_arg = arg.parse_args()

    local_owl = LocalKOwl()

    image_list = move_image(sys_arg.generate_save_dir, sys_arg.generate_output_dir)
    output_path = os.path.join(sys_arg.generate_output_dir, "Annotation")

    create_check_exists_file(output_path)

    result_label = []
    result_annotation = []
    for i, image_path in tqdm(
            enumerate(image_list),
            desc="Annotating images",
            total=len(image_list),
    ):
        image = Image.open(image_path)
        processed_text, entities = local_owl.annotate(image)
        print(processed_text)

        print(entities)
        create_xml_file(image_id=image_path.rsplit("/", 1)[-1],
                        content=entities,
                        image_size=image.size,
                        output_path=output_path,
                        label_list=result_label)
        #create_yoyo_format(image_id=image_path.rsplit("/", 1)[-1],content=entities, image_size=image.size,output_path=output_path, label_list=result_label)


    with open(os.path.join(output_path, "class.txt"), "w") as f:
        f.write("\n".join(set(result_label)))

    
        # print(processed_text)
        # `An image of a snowman warming himself by a fire.`

        # print(entities)
        # `[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]`

