import os
import cv2
import json
import torch
import torchvision
import numpy as np
from glob import glob


def get_boxes_from_yolo(file_path,image_shape, get_scores=False):
    '''
    Read the yolo labels from the given file, convert to xxyy format and return
    '''
    with open(file_path,"r") as f:
        lines = [line.replace("\n","").split() for line in f.readlines()]
    
    res=[]
    scores = []
    for line in lines:
        line = [float(i) for i in line]
        xmin = (line[1]-line[3]/2)*image_shape[1]
        xmax = (line[1]+line[3]/2)*image_shape[1]
        ymin = (line[2]-line[4]/2)*image_shape[0]
        ymax = (line[2]+line[4]/2)*image_shape[0]
        if get_scores:
            scores.append(line[5])
        res.append((xmin,ymin,xmax,ymax))
    
    res = np.array(res).astype(np.int16)
    if get_scores:
        return res, np.array(scores)
    
    return res

def get_boxes_from_yolo_folder(folder_path,image_shape):
    label_paths = glob(os.path.join(folder_path,"*.txt"))
    if len(label_paths)==0:
        raise ValueError(f"{folder_path} does not contain any .txt files")
    
    all_boxes = []

    for path in label_paths:
        boxes= get_boxes_from_yolo(path,image_shape)
        all_boxes.extend(boxes)
   
    all_boxes = torch.from_numpy(np.array(all_boxes)).to(int)

    return all_boxes
    
    
def rectangles_on_mip(mip, boxes, box_format="xxyy_sep", color=(0,0,255), thickness=1, show_indices=True, text_color=(255,255,255), text_size=0.3):
    '''
    A function for ploting rectangles on a picture, returns the picture\n
    Parameters:
        - `mip`: The picture(ndarray) to which the rectangles will be applied to
        - `boxes`: list of boxes, which will be applied on the image
        - `box_format`: the format of the boxes which are given to the function
        Accepts one of the following values:
            - `xyxy_sep`: if the boxes are given in the following format `([xmins...], [ymins...], [xmaxs...], [ymaxs...])` |-> each coordinates of boxes are in the same lists
            - `xxyy_sep`: if the boxes are given in the following format `([xmins...], [xmaxs...], [ymins...], [ymaxs...])` |
            - `xyxy_one`: if the boxes are given in the following format `[xmin, ymin, xmax, ymax], ...` |-> each box is separate
            - `xxyy_one`: if the boxes are given in the following format `[xmin, xmax, ymin, ymax], ...` |
            - `xywh`: if the boxes are given in the following format `(xcenter, ycenter, width, height)` 
            - `pt1,pt2`: if the boxes are given in the following format `((xmin, ymin), (xmax, ymax))`
        - `color`: an RGB `color(R,G,B)` : `R,G,B <-- (0,255)`
    '''
    box_format = box_format.lower()
    im = mip.copy()
    if box_format=="xyxy_sep":
        boxes_for_plotting = [((boxes[0][i],boxes[1][i]),(boxes[2][i],boxes[3][i])) for i in range(len(boxes[0]))]
    elif box_format=="xxyy_sep":
        boxes_for_plotting = [((boxes[0][i],boxes[2][i]),(boxes[1][i],boxes[3][i])) for i in range(len(boxes[0]))]
    elif box_format=="xyxy_one":
        boxes_for_plotting = [((boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3])) for i in range(len(boxes))]
    elif box_format=="xxyy_one":
        boxes_for_plotting = [((boxes[i][0],boxes[i][2]),(boxes[i][1],boxes[i][3])) for i in range(len(boxes))]
    elif box_format=="pt1,pt2":
        boxes_for_plotting = boxes
    else:
        raise ValueError(f'`box_format` should be one of the following ["cr","xyxy_sep","xxyy_sep","xyxy_one","xxyy_one", "pt1,pt2"] but got {box_format}')
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i,box in enumerate(boxes_for_plotting):
        cv2.rectangle(im, box[0], box[1], color=color, thickness=1)
        if (show_indices):
            im = cv2.putText(im, f"{i}", box[0], font, text_size, text_color, 1, cv2.LINE_AA)    
    return im


def get_boxes_from_json_files(root_path, nms_threshold=0.1):
    # Get json filenames
    if root_path[-5:]!=".json":
        files = glob(os.path.join(root_path,"*.json"))
    else:
        files = [root_path]
    frames = []
    # Read all json files in root_path
    for file in files:
        with open(file, "r") as f:
            labels = json.load(f)
            frames.append([item["points"] for item in labels["shapes"]])
    
    all_boxes = []
    for index, _ in enumerate(frames):
        boxes = []
        for jindex, _ in enumerate(frames[index]):
            boxes.append(frames[index][jindex][0][:])
            boxes[-1].extend(frames[index][jindex][1][:])
        if boxes == []:
            continue
    
        all_boxes.append(boxes[:])
    
    boxes_per_frames = []
    for frame in all_boxes:
        boxes_per_frames.append(torch.tensor(frame))
        
    resulting_boxes = boxes_per_frames[0]
    for boxes in boxes_per_frames[1:]:
        if boxes.ndim == 1:
            boxes = boxes[None, ...]
        iou_matrix = torchvision.ops.box_iou(resulting_boxes, boxes)
        new_box_indices = torch.where(iou_matrix.sum(dim=0)< nms_threshold)
        resulting_boxes = torch.cat((resulting_boxes, boxes[new_box_indices])) 
    
    return resulting_boxes[:,[0,2,1,3]]


class Metrics:
    def box_iom(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        area1 = torchvision.ops.box_area(boxes1)
        area2 = torchvision.ops.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        min_area = torch.min(area1[:,None,...],area2)

        return inter/min_area


    def compare_boxes(self, true_boxes, pred_boxes, print_=False, dec_p=4, confusion_matrix=False):
        iom = self.box_iom(true_boxes,pred_boxes)

        tp = 0
        fn = (iom.sum(dim=1)==0).sum()
        fp = (iom.sum(dim=0)==0).sum()

        # for each gt box check
        for i, row in enumerate(iom):
            if row.sum()!=0:
                col=row.argmax()
                iom[:,col]=0
                fn += (iom[i:, iom[i]>0]>0).sum()==1
                tp+=1

        fp += (iom.sum(dim=0)>0).sum()


        Precision = (tp/(tp+fp)).item()
        Recall = (tp/(tp+fn)).item()
        F1 = (2*Precision*Recall/(Precision+Recall))

        if print_:
            if confusion_matrix:
                print(f"TP={tp}, FN={fn}, TN={tn}, FN={fp}\n")
            print(f"{'Precision:':<11} {round(Precision, dec_p)}\n{'Recall:':<11} {round(Recall, dec_p)}\n{'F1:':<11} {round(F1, dec_p)}")
        return Precision, Recall, F1, (tp,fn,fp)


    def pretty_print(self, tens):
        print("   ", end="")
        for i in range(tens.shape[1]):
            print(f"{i:<3}| ",end="")
        print("\n","-"*70)
        for r,i in enumerate(tens):
            print(f"{r})", end = " ")
            for j in i:
                print(f"{j.item():.2f}", end=" ")
            print("\n")
            
def print_stats(true_boxes, pred_boxes, times=None):
    Metric = Metrics()



    tp = Metric.compare_boxes(true_boxes, pred_boxes, print_=True)[-1][0]
    print(f"\nThere are {true_boxes.shape[0]} bacteria in the folder")
    print(f"Correctly detected {tp} bacteria out of {pred_boxes.shape[0]} predictions\n")
    
    
    if times:
        print(f"{'Average time for finding the start frame':<47} --- {np.mean(times['finding_start_frame']):.2f} sec. | {len(times['finding_start_frame'])} folder(s)")
        print(f"{'Average time for single frame inference':<47} --- {np.mean(times['inference_time']):.2f} sec. | {len(times['inference_time'])} image(s)")
        print(f"{'Average time for preprocessing a single frame':<47} --- {np.mean(times['preprocess_time']):.2f} sec. | {len(times['preprocess_time'])} image(s)")
        print(f"{'Average time for postprocessing a single frame':<47} --- {np.mean(times['postprocess_time']):.2f} sec. | {len(times['postprocess_time'])} image(s)")
        print(f"{'Average total time for detection':<47} --- {np.mean(times['total_time']):.2f} sec. | {len(times['total_time'])} folder(s)")
        

