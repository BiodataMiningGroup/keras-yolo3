import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import ast

""" Reads a list of images from the image_list_file and passes
    them to the network yolo for prediction.

    Images are loaded from the image_directory.
"""
def detect_img(yolo, image_list_file, image_directory):
    f = open(image_list_file,"r")
    images = ast.literal_eval(f.read())
    print(images)
#    images = ['777226.png', '777242.png', '777259.png', '777275.png', '777292.png', '777308.png', '783297.png', '783355.png', '783517.png', '783963.png', '784451.png', '784599.png', '786203.png', '786236.png']
#    images = ['1.png','39.png','678.png','234.png','55.png','100.png','101.png','200.png']
    for img in images:
        print(os.getcwd(), img)
        try:
            image = Image.open(image_directory+img)
        except Exception as e:
            print(e)
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save("result_image_"+img)
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
        detect_img(YOLO(**vars(FLAGS)), FLAGS.image_list_file, FLAGS.image_directory)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
