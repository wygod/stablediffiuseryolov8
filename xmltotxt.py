#-*- encoding:utf-8 -*-

import argparse
import os.path
import xml
import xml.dom.minidom as minidom
from xml.etree import ElementTree as et


from xml.dom.minidom import parse


def have_label(name, self_label):
    return [i_label for i_label in self_label if i_label in name]


def xml2txt(xml_path, output_path, base_label):
    DOMTree = xml.dom.minidom.parse(xml_path)
    collection = DOMTree.documentElement
    image_size = collection.getElementsByTagName("size")
    width = image_size[0].getElementsByTagName("width")[0].childNodes[0].data
    height = image_size[0].getElementsByTagName("height")[0].childNodes[0].data

    content = collection.getElementsByTagName("object")
    label_source = []
    for i_content in content:
        temp = []
        label = i_content.getElementsByTagName("name")[0].childNodes[0].data
        if label:
            label_value = have_label(label, base_label)
            if label_value:
                size_value = i_content.getElementsByTagName("bndbox")
                x_min = int(size_value[0].getElementsByTagName("xmin")[0].childNodes[0].data)
                y_min = int(size_value[0].getElementsByTagName('ymin')[0].childNodes[0].data) / float(height)
                x_max = int(size_value[0].getElementsByTagName('xmax')[0].childNodes[0].data) / float(width)
                y_max = int(size_value[0].getElementsByTagName('ymax')[0].childNodes[0].data) / float(height)
                temp.append(str((x_min + x_max) / 2 / float(width)))
                temp.append(str((y_min + y_max) / 2 / float(height)))
                temp.append(str((x_max - x_min)/float(width)))
                temp.append(str((y_max - y_min)/float(height)))
                label_source.append(str(base_label.index(label_value[0])) + " "+ " ".join(temp))

    xml_path_clear =xml_path.rsplit("\\", 1)[-1] if "\\" in xml_path else xml_path

    with open(os.path.join(output_path, xml_path_clear.replace("xml", "txt")).replace("/","\\"), 'w') as f:
        f.write("\n".join(label_source))


def xml2txt2file(input_path, output_path, base_list):
    if os.path.isdir(input_path):
        list_file = os.listdir(input_path)
        for i_file in list_file:
            if i_file.endswith("xml"):
                xml2txt(os.path.join(input_path, i_file).replace("/", "\\"), output_path, base_list)
    elif os.path.isfile(input_path):
        xml2txt(input_path, output_path, base_list)
    else:
        raise Exception("input file error")


if __name__=="__main__":
    meta_label = ['car', 'truck', 'motorcycle']
    xml2txt2file('C:\\Users\\Yi\\Desktop\\fsdownload\\car_trcuk\\testimage\\Annotation',
                 'C:\\Users\\Yi\\Desktop\\fsdownload\\car_trcuk\\testimage\\Annotationtxt',meta_label)
    # doc = et.parse(xml_path)
    # elem = doc.documentElement
    # width = elem.getElementsByTagName('width')

    # height = elem.getElementsByTagName('height')
    #
    # object_class = elem.getElementsByTagName('object')
    #
    # for i in object_class:
    #
    #
    # height = elem.getElementsByTagName('object')
    #
    #
    # doc


