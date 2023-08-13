import os
import xml.etree.ElementTree as ET
from PIL import Image

def save_bounding_box_images(image_path, xml_path, output_folder):
    # Load the image
    image = Image.open(image_path)
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Iterate through each object in the XML
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        # Crop and save the bounding box image
        box_image = image.crop((xmin, ymin, xmax, ymax))
        output_path = os.path.join(output_folder, f'{name}_{xmin}_{ymin}_{xmax}_{ymax}.jpg')
        box_image.save(output_path)
    
    print("Bounding box images saved successfully!")


image_path = 'C:\\Users\\Prakash Gautam\\Desktop\\Child_Growth_Monitor\\test\\2023-05-29_IMG_1685356462189.jpg'
xml_path = 'C:\\Users\\Prakash Gautam\\Desktop\\Child_Growth_Monitor\\xml_files\\2023-05-29_IMG_1685356462189.xml'
output_path = 'image_boundig'

save_bounding_box_images(image_path, xml_path, output_path)