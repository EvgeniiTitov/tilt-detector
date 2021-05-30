### Inclination calculator


Two ways to run the script:

1. 

- Clone the repository

- Install the prod requirements

- Download model weights from https://drive.google.com/file/d/1NmAHAVGirCVn6uzO9UkDj70sGqXFK0o0/view?usp=sharing

- Put them under object_detector/yolov4/dependencies

Usage:
```
usage: main.py [-h] [--path_to_folder PATH_TO_FOLDER] [--destination DESTINATION] [--results_to_json]

optional arguments:
  -h, --help            show this help message and exit
  --path_to_folder PATH_TO_FOLDER
                        Path to a folder containing images to process
  --destination DESTINATION
                        Path to a folder where processed images will be saved
  --results_to_json     Save processing results as a JSON file
  
Example:
python main.py 
python main.py --results_to_json
python main.py --destination path/to/destination/folder
python main.py --path_to_folder path/to/images/to/process
```

---
2. Pull docker image from LINK - TBA