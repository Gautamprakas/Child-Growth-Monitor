from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
# Import packages
import time
import os
import cv2
import numpy as np
from xml.dom import minidom
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:\\Users\Prakash Gautam\\Desktop\\Child_Growth_Monitor\\test'  # Folder to store uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def text_to_xml(txt_path, xml_output_path):
    # Read the text file
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()

    # Create a dictionary to store object information based on class name
    object_info = {}

    for line in lines:
        parts = line.strip().split()
        object_name = parts[0]
        confidence = float(parts[1])
        xmin = int(parts[2])
        ymin = int(parts[3])
        xmax = int(parts[4])
        ymax = int(parts[5])
        
        if object_name not in object_info or confidence > object_info[object_name]['confidence']:
            object_info[object_name] = {
                'confidence': confidence,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }

    # Extract information from the dictionary and create XML
    xml_annotation = '<annotation>\n'
    xml_annotation += f'\t<folder>All Images</folder>\n'
    
    # Extract the base file name without extension
    base_filename = os.path.splitext(os.path.basename(txt_path))[0]
    
    # Get the full absolute path of the input text file
    full_abs_path = os.path.abspath(txt_path)
    
    xml_annotation += f'\t<filename>{base_filename}.jpg</filename>\n'
    xml_annotation += f'\t<path>{full_abs_path.replace(".txt", ".jpg")}</path>\n'
    
    for object_name, info in object_info.items():
        xml_annotation += '\t<object>\n'
        xml_annotation += f'\t\t<name>{object_name}</name>\n'
        xml_annotation += f'\t\t<confidence>{info["confidence"]}</confidence>\n'
        xml_annotation += '\t\t<bndbox>\n'
        xml_annotation += f'\t\t\t<xmin>{info["xmin"]}</xmin>\n'
        xml_annotation += f'\t\t\t<ymin>{info["ymin"]}</ymin>\n'
        xml_annotation += f'\t\t\t<xmax>{info["xmax"]}</xmax>\n'
        xml_annotation += f'\t\t\t<ymax>{info["ymax"]}</ymax>\n'
        xml_annotation += '\t\t</bndbox>\n'
        xml_annotation += '\t</object>\n'
    xml_annotation += '</annotation>'

    # Write the XML to the output file
    with open(xml_output_path, 'w') as xml_file:
        xml_file.write(xml_annotation)

def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='result', txt_only=True):

  # Grab filenames of all images in test folder
  images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

  # Load the label map into memory
  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)

  input_mean = 127.5
  input_std = 127.5

  # Randomly select test images
  images_to_test = random.sample(images, num_test_images)

  # Loop over every image and perform detection
  for image_path in images_to_test:

      # Load image and resize to expected shape [1xHxWx3]
      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      imH, imW, _ = image.shape
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      detections = []

      # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))

              cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])


      # All the results have been drawn on the image, now display the image
      if txt_only == True: # "text_only" controls whether we want to display the image results or just save them in .txt files
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite("image_box/first.jpg",image)
        
        # plt.figure(figsize=(12,16))
        # plt.imshow(image)
        # plt.show()

      # Save detection results in .txt files (for calculating mAP)
      if txt_only == True:

        # Get filenames and paths
        image_fn = os.path.basename(image_path)
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
                
  return
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image part', 400
        image_file = request.files['image']
        if image_file.filename == '':
            return 'No selected file', 400
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            # Call the tflite_detect_images function on the uploaded image
            tflite_detect_images(PATH_TO_MODEL, app.config['UPLOAD_FOLDER'], PATH_TO_LABELS, min_conf_threshold, images_to_test, txt_only=True)
            time.sleep(15)
            # Get the path of the generated text file
            txt_path = os.path.join("result", os.path.splitext(filename)[0] + '.txt')

            # Call the text_to_xml function on the generated text file
            xml_output_name = os.path.splitext(filename)[0] + '.xml'
            xml_output_directory = 'xml_files'
            xml_output_path = os.path.join(xml_output_directory, xml_output_name)
            text_to_xml(txt_path, xml_output_path)
            uploaded_image_name = filename
            # Read the generated XML file to extract object dimensions
            xml_dom = minidom.parse(xml_output_path)
            objects = xml_dom.getElementsByTagName("object")

            dimensions_info = []

            for obj in objects:
                name = obj.getElementsByTagName("name")[0].firstChild.data
                bndbox = obj.getElementsByTagName("bndbox")[0]
                xmin = int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
                ymin = int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
                xmax = int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
                ymax = int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
                dimensions = {
                "name": name,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax
            }
                dimensions_info.append(dimensions)

            image_url = f"/{app.config['UPLOAD_FOLDER']}/{filename}"
            return render_template('index.html', image_url=image_url, dimensions_info=dimensions_info, uploaded_image_name=uploaded_image_name)


            

    return render_template('index.html')

if __name__ == '__main__':
    # Set up variables for running user's model
    PATH_TO_IMAGES = 'test'
    PATH_TO_MODEL = 'custom_model_lite/detect.tflite'
    PATH_TO_LABELS = 'labelmap.txt'
    min_conf_threshold = 0.1
    images_to_test = 1

    # Create the XML output directory if it doesn't exist
    xml_output_directory = 'xml_files'
    if not os.path.exists(xml_output_directory):
        os.makedirs(xml_output_directory)

    # Run the application
    app.run(debug=True, port=8000)

