import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import numpy as np
import ast
import evaluate

""" Reads the annotation file and
    returns a dict with the filenames as keys and lists
    of boxes as values.
"""
def parse_annotation_file(eval_fn):
    boxes_dict = {}
    with open(eval_fn) as f:
        lines = f.readlines()
        for l in lines:
            l = l.split()
            boxes = np.array([np.array(list(map(int,box.split(',')))) for box in l[1:]])
            boxes_dict[l[0]] = boxes
    f.close()
    return boxes_dict


""" Reads a list of images from the image_list_file and passes
    them to the network yolo for prediction.

    Images are loaded from the image_directory.
"""
def detect_img(yolo, FLAGS):

    #TODO read annotation file if given and calculate iou of boxes
    f = open(FLAGS.image_list_file,"r")
    images = ast.literal_eval(f.read())
    print(images)
    f.close()

    if FLAGS.evaluation_file:
        boxes_dict = parse_annotation_file(FLAGS.evaluation_file)

    for img in images:
        print(os.getcwd(), img)
        try:
            image = Image.open(FLAGS.image_directory+img)
        except Exception as e:
            print(e)
            print('Open Error! Try again!')
            continue
        else:
            r_image, r_boxes, r_classes = yolo.detect_image(image)
            r_image.save("result_image_"+img)
            if FLAGS.evaluation_file:
                true_boxes = boxes_dict[FLAGS.image_directory+img]
                true_boxes = [[int(true_box[1]), int(true_box[0]), int(true_box[3]), int(true_box[2])] for true_box in true_boxes]
                #TODO to int or not?
#                r_boxes = [[int(r_box[1]), int(r_box[0]), int(r_box[3]), int(r_box[2])] for r_box in r_boxes]
                print("True boxes", true_boxes)
                print("Result boxes", r_boxes)
                print(evaluate.countHits(r_boxes, true_boxes, thresh=0.3))
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image_list_file', type=str,
        #TODO help
    )

    parser.add_argument(
        '--image_directory', type=str,
        #TODO help
    )

    parser.add_argument(
        '--evaluation_file', type=str, default=None,
        #TODO help
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)), FLAGS)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
