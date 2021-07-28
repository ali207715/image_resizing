#!/usr/bin/env python

import cv2 as cv
import sys
import json
import numpy as np

user_input = list(sys.argv)
source_file = user_input[1]
annotation_file = user_input[2]
dimensions = (int(user_input[3]), int(user_input[4]))  # The intended resolution (width, height)

img = cv.imread(source_file)
height, width, extra = img.shape
width_diff = dimensions[0] / width  # The factor by which to move each point in the annotations.
height_diff = dimensions[1] / height

annotation_create = {}  # Dict to store new data for easy annotation on the image.

resized_img = cv.resize(img, dimensions)

# Modifying the json file
with open(annotation_file, "r", encoding="utf-8") as data:
    annotation_data = json.load(data)
    data.close()

for dat in annotation_data['segmentedObjectDict']:
    Caption = annotation_data['segmentedObjectDict'][dat]["Name"]
    fontsize = annotation_data['segmentedObjectDict'][dat]["fontSize"]
    for pts in annotation_data['segmentedObjectDict'][dat]["pointsList"]:
        pts[0] *= width_diff
        pts[1] *= height_diff
    Points = annotation_data['segmentedObjectDict'][dat]["pointsList"]
    annotation_create[(Caption, fontsize)] = Points

# ------------ OUTPUT ----------------------------------------------------------------------------------------- #
# JSON DUMP
output = open("output_json.json", "w")
json.dump(annotation_data, output)
output.close()

# RESIZED IMAGE DUMP
cv.imwrite("output_resized_img.jpg", resized_img)

# ANNOTATED IMAGE
# Specific font size is available but font type and color is not specified, hence choosing randomly.
# Font size provided used but is too small to be visible, hence using value of 0.20. If need to be changed then access
# id[1] for usage in the loop below.

for id, pts in annotation_create.items():
    pts_array = np.array(pts)
    resized_img = cv.polylines(resized_img, np.int32([pts_array]), True, (255, 255, 255), thickness=1)
    resized_img = cv.putText(resized_img, id[0], (int(pts[-1][0]), int(pts[-1][1])), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.2, color=(255, 255, 255))

cv.imwrite("output_annotated_img.jpg", resized_img)
