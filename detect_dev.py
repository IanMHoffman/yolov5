import argparse
import os
import re
import platform
import shutil
import time
import sys
import tqdm
import glob
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':

    # Creates a new folder for the output so multiple instances can be run at once
    folder_time = str(time.strftime("%a-%d-%b-%Y-%H-%M-%S")) # Mon-1-Jan-2020-00-00-00
    output_folder = r'\output-' + folder_time
    output_path = r'C:\YOLOv5\yolov5\inference' + output_folder
    print("Output Folder: " + output_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\YOLOv5\yolov5\weights\last.pt', help='model.pt path(s)') # defaults to trained weights
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=output_path, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt') # store false defaults it to saving text files
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    start_time = time.time()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

    path = Path(output_path)

    img_path = Path(opt.source)

    print(path)

    if not os.path.exists((path.joinpath("cropped"))):
        os.makedirs((path.joinpath("cropped")))

    cropped_path = Path(path, 'cropped')

    def isBigger (biggest, current):
        if biggest[3] > current[3] and biggest[4] > current[4]:
            return False
        else:
            if biggest[3] * biggest[4] > current[3] * current[4]: # area check
                return False
            return True

    
    # Find the root path of the Section
    sectionPath = img_path
    while re.match(r'Section', sectionPath.stem, re.IGNORECASE) is None:
        sectionPath = sectionPath.parent
    # Join Referenced Images to Section root path
    referencedPath = sectionPath.joinpath("ReferencedImages")

     # Find all of the original images and copy them to referenced
    origList = list(img_path.glob( '*.jpg' ))

    # Copies the original images to referenced images overwriting the existing files
    for original in tqdm.tqdm(origList, desc="Copying Holding to Referenced"):
        shutil.copy(Path(original), referencedPath)

    # Find all of the text files that were detected
    detectList = list(path.glob( '*.txt' ))

    for file in tqdm.tqdm(detectList, desc="Cropping, Resizing & Saving Images"):
        with file.open() as f:
            biggest = (0,0,0,0,0)

            for line in f:
                temp = tuple(map(float,line.split())) # typcast to a float
                if isBigger(biggest, temp):
                    biggest = temp # if returned true temp is now the biggest
            biggest = biggest + (file.stem,) # append the file name without extension to the tuple
            
            # read image
            img = cv2.imread(str(img_path.joinpath(str(biggest[5]) + '.jpg')))
            dimensions = img.shape # return size of image (height, width, channels) note: opencv is y,x for most things

            # denormalize the coridantes for this image
            x_cen = int(biggest[1] * dimensions[1]) # normalized x-cen * image width
            y_cen = int(biggest[2] * dimensions[0]) # normalized y-cen * image height
            width = int(biggest[3] * dimensions[1]) # normalized width * image width
            height = int(biggest[4] * dimensions[0]) # normalized height * image height

            # This is the border around the image in pixels after the crop
            padding = 30
            botttom_padding =  30

            x_start = x_cen - int(width / 2) - padding
            x_end = x_cen + int(width / 2) + padding 

            y_start = y_cen - int(height / 2) - padding 
            y_end = y_cen + int(height / 2) + padding + botttom_padding # bottom

            if x_start < 0:
                x_start = 0
            if x_end > dimensions[1]:
                x_end = dimensions[1] - 1
            if y_start < 0:
                y_start = 0
            if y_end > dimensions[0]:
                y_end = dimensions[0] - 1

            cropped = img[y_start:y_end, x_start:x_end].copy() # y,x

            #cv2.imwrite(str(cropped_path.joinpath(str(biggest[5]) + '.jpg')) , cropped)

            # default export horizontal (x axis) size that gets the jpegs into the required 300-600kb range
            desiredWidth = 1300
            scale_percent = int( desiredWidth / cropped.shape[1]) # percent of original width size
            # resize image
            resized = cv2.resize(cropped, None, fx=scale_percent, fy=scale_percent, interpolation = cv2.INTER_AREA)

            resized_path = str(cropped_path.joinpath(str(biggest[5]) + '.jpg'))

            # default export quality that gets the jpegs into the required 300-600kb range
            export_quality = 90

            cv2.imwrite(resized_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), export_quality])

            # if the files are to smaller than 400kb (400,000b) make them bigger
            while os.stat(resized_path).st_size < 400000: # setting to 300kb creates files that are 262kb
                if export_quality >= 100:
                    desiredWidth = desiredWidth + 100
                    scale_percent = int( desiredWidth / cropped.shape[1]) # percent of original width size
                    resized = cv2.resize(cropped, None, fx=scale_percent, fy=scale_percent, interpolation = cv2.INTER_AREA)
                export_quality = export_quality + 1
                cv2.imwrite(resized_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), export_quality])
            
            # if the files are to bigger than 600kb (600,000b) make them smaller
            while os.stat(resized_path).st_size > 600000:
                if export_quality <= 80:
                    desiredWidth = desiredWidth - 100
                    scale_percent = int( desiredWidth / cropped.shape[1]) # percent of original width size
                    resized = cv2.resize(cropped, None, fx=scale_percent, fy=scale_percent, interpolation = cv2.INTER_AREA)
                export_quality = export_quality - 1
                cv2.imwrite(resized_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), export_quality])
    
    # Find all of the cropped images and move them
    croppedList = list(cropped_path.glob( '*.jpg' ))

    # Copies the processed images to the original location overwriting the existing files
    for croppedImg in tqdm.tqdm(croppedList, desc="Copying Processed Images to Holding"):
        shutil.copy(Path(croppedImg), img_path)

    # Delete the folder that was just created to hold images
    shutil.rmtree(output_path)

# C:\YOLOv5\yolov5\Scripts\python.exe C:\YOLOv5\yolov5\detect.py --source 'Section0'