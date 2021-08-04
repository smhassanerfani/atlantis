#!/usr/bin/python
# Luis Baez - XML Parsing

import sys
import getopt
import copy
import xml.etree.ElementTree as ET
from xml.dom import minidom
from io import BytesIO


def main(argv):
    new_xml = ''
    old_xml = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('xml_extractor.py -i <old .xml file> -o <new .xml file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('xml_extractor.py -i <old XML file> -o <new XML file>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            old_xml = arg
        elif opt in ("-o", "--ofile"):
            new_xml = arg

    parse(new_xml, old_xml)


def parse(new_xml, old_xml):

    # 1. Get unique file name from old tree
    # 2. Find unique file name in new tree
    # 3. If 2. is true, copy sub-trees (child trees) from old to new

    old_tree = ET.parse(old_xml)
    new_tree = ET.parse(new_xml)
    old_root = old_tree.getroot()
    new_root = new_tree.getroot()

    for image1 in old_root.findall('image'):
        img_n1 = image1.get('name')
        # got image name, now check for image name in new file

        for image2 in new_root.findall('image'):
            img_n2 = image2.get('name')

            if(img_n1 == img_n2):
                # Found matching image names, now copy img_n1 child to img_n2
                for poly in image1:
                    # Grabbed polygon element, now paste to new_tree after img_n2
                    image2.append(poly)

    # Write the new_tree to file. Include Encoding and xml_declaration
    new_tree.write("finalized_ext.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main(sys.argv[1:])
