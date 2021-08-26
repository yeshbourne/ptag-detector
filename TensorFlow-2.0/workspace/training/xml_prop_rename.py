import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

path = "./images/xml"

for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tree.find('filename').text = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
        #print(tree.find('filename').text)
        tree.write(xml_file)