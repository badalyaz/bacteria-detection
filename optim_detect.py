import os
import sys
import cv2
import pickle
import ctypes
import argparse
import threading
import numpy as np
from glob import glob
from time import time
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering as AGC

def rectangles_on_mip(mip, boxes, box_format="xxyy_one", color=(255,0,0), thickness=1, show_indices=True, text_color=(255,255,255), text_size=0.3):
    '''
    A function for ploting rectangles on a picture, returns the picture\n
    Parameters:
        - `mip`:The picture(ndarray) to which the rectangles will be applied to
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

LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_file_path, conf_thres=0.4, iou_thres=0.5, save_results=False, out_path="./", agc_distance=20):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        self.runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())


        self.context = engine.create_execution_context()
       
        self.image_cache = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store variables
        # FOR INFERENCE
        self.stream = stream
        self.engine = engine

        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agc_distance = agc_distance
        
        self.empty_image_found = False
        self.ready_to_read = True

        self.save_results = save_results
        self.out_path = out_path
        self.color = {'white':      "\033[1;37m",
                      'yellow':     "\033[1;33m",
                      'green':      "\033[1;32m",
                      'blue':       "\033[1;34m",
                      'cyan':       "\033[1;36m",
                      'red':        "\033[1;31m",
                      'magenta':    "\033[1;35m",
                      'black':      "\033[1;30m",
                      'darkwhite':  "\033[0;37m",
                      'darkyellow': "\033[0;33m",
                      'darkgreen':  "\033[0;32m",
                      'darkblue':   "\033[0;34m",
                      'darkcyan':   "\033[0;36m",
                      'darkred':    "\033[0;31m",
                      'darkmagenta':"\033[0;35m",
                      'darkblack':  "\033[0;30m",
                      'off':        "\033[0;0m"}
 
        if self.save_results:
            self.times = {
                "finding_start_frame" : [],
                "inference_time" : [],
                "preprocess_time" : [],
                "postprocess_time" : [],
                "total_time" : []
            }
        
    def contains_image(self, folder_path):
        if len(glob(os.path.join(folder_path,'*.jpeg'))):
            return True
        else:
            return False
    def get_folder_paths(self, root_path):
        if self.contains_image(root_path):
            return [root_path]
        else:
            image_folders = []
            for folder in glob(os.path.join(root_path,"*")):
                if self.contains_image(folder):
                    image_folders.append(folder)

            if len(image_folders)==0:
                raise FileNotFoundError("Check the given path, no images found in depth=2")

            return image_folders
    def filter_bboxes(self, bboxes):   
        centers_x = (np.mean((bboxes[:, 2], bboxes[:, 0]), axis=0))[:, None]
        centers_y = (np.mean((bboxes[:, 3], bboxes[:, 1]), axis=0))[:, None]
        centers = np.concatenate((centers_x, centers_y), axis=1)
        
        AGC_ = AGC(n_clusters=None, distance_threshold=self.agc_distance)
        clustered=(AGC_.fit_predict(centers))

        uniques_ = np.unique(clustered)
        filtered_boxes = np.empty((0,7))
        for unique in uniques_:
            area_boxes = bboxes[ np.where(clustered==unique)[0] ]
            unique_frames = np.unique(area_boxes[:,-1],return_counts=True)
            # If the the area contains a single box, select it according to its confidence, otherwise
            # Select the boxes that are detected in a single frame
            if unique_frames[1].max()==1:
                filtered_boxes = np.concatenate((filtered_boxes, area_boxes[None,np.argmax(area_boxes[:,-3])]),axis=0)
            else:
                filtered_boxes = np.concatenate((filtered_boxes, area_boxes[area_boxes[:,-1]==unique_frames[0][np.argmax(unique_frames[1])]]),axis=0)

        return filtered_boxes
    def preprocess(self, img_path, img_ind):
        image = cv2.imread(img_path)
        h, w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        # Normalize to [0,1]
        image = image.astype(np.float32)/255.0
        # HWC to NCHW format
        # np.ascontiguousarray --> Convert the image to row-major order, also known as "C order":
        image =  np.ascontiguousarray(image.transpose([2,0,1])[None])

        return image.ravel(), h, w, img_ind
    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=self.conf_thres, nms_thres=self.iou_thres)

        return boxes
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > self.conf_thres
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
        
    def reading_thread(self, full_image_paths, top_ind, direction):
        # A loop for checking wether an emty image was found
        while not self.empty_image_found:
            # When starting, the second while loop will help to preload 2 images
            while len(self.image_cache) < 2 and self.ready_to_read:
                top_ind += direction
                print(full_image_paths[top_ind])
                self.image_cache.append(self.preprocess(full_image_paths[top_ind], top_ind))
        self.image_cache.clear()        
    def run_engine(self):
        if len(self.image_cache) == 0:
            return

        image, h, w, frame_ind = self.image_cache[0]

        self.ctx.push()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], image, self.stream)
        # Run inference.
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

        del self.image_cache[0]
        preds = self.post_process(self.host_outputs[0][:LEN_ALL_RESULT], h, w)

        if preds.shape[0] == 0:
            self.empty_image_found = True
            self.image_cache.clear()
            return

        if len(preds.shape) == 1:
            preds = preds[None]

        preds = np.concatenate([preds, np.full((preds.shape[0], 1), frame_ind)], axis=-1)
        self.res_cache = np.concatenate((self.res_cache, preds))
        # print(self.res_cache.shape)


    def up_and_down(self, full_image_paths, top_ind, direction):
        '''
        A method for reading images in 2 directions from given starting point
        
        Inputs:
            - `image_paths`: List of paths for .jpeg images in a single folder
            - `idx`: The starting index of the method
            - `direction`: Defines the direction of path reading
                - `1`: The function reads images from idx to right
                - `-1`: The functions reads images from idx to left
        Output: None
        '''
        thread_reading = threading.Thread(target=self.reading_thread, args=[full_image_paths, top_ind, direction])
        thread_reading.start()
        while not self.empty_image_found:
            self.run_engine()
        thread_reading.join()
            
    def select_top_indices(self, weights, count):
        splits = np.linspace(0,len(weights),count//4, dtype=int)
        indices = np.argsort(-weights)
        indices = indices[splits[:-1]+np.diff(splits)//2]
        return indices[indices<count]
    def get_image_paths(self,folder_path):
        image_paths = list(filter(lambda x: x, list(glob(folder_path+"/*.jpeg") + glob(folder_path+"/*/*.jpeg"))))[:-1]
        image_paths.sort(key=lambda x: x[x.rfind("/"):])
        
        return image_paths
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        del self.context

    def infer(self, root_path, frame_weights):
        start = time()
        # Getting the folders which contain images 
        folder_paths = self.get_folder_paths(root_path)

        # The final results for each folder will be stored in self.results
        self.results = []
        for path in folder_paths:
            st = time()
            # Getting the image paths in the folder
            print(f"\n{self.color['green']}Processing {self.color['darkcyan']}{path} {self.color['green']}directory\n")
            
            # # Creating an empty ndarray for storing each folders detections in it
            # self.mip_cache = np.zeros((self.input_w, self.input_h, 3))
            self.res_cache = np.empty((0,7))
            
            # Getting the image paths in the given directory
            image_paths = self.get_image_paths(path)
            # Loading the frame weights and selecting the top N out of those, where N is the number of frames
            selected_idxs = self.select_top_indices(weights=np.load(frame_weights), count=len(image_paths))
            i = selected_idxs[0]
            self.image_cache.append(self.preprocess(image_paths[i], i))
            self.run_engine()
            # if self.save_results:
            #     self.times["finding_start_frame"].append(time())
            # Checking frames with best weights
            # for i in selected_idxs:
            #     # Getting the starting index for multi-threaded detection(Up and Down)
            #     self.res_cache = self.read_and_detect(image_paths[i])
            #     # self.mip_cache = self.image_cache
            #     if self.res_cache.shape[0]:
            #         self.res_cache = np.concatenate([self.res_cache, np.full((self.res_cache.shape[0], 1), i)], axis=-1)
            #         break
            # if self.res_cache.shape[0]==0:
            #     print(f"Detected {self.res_cache.shape[0]} Bacteria in {path}\nDetection time >> {(time()-st):.2f} sec.\n")
            #     print(f"{self.color['green']}{'-'*40}{self.color['off']}")
            #     continue
                
            # if self.save_results:
            #     self.times["finding_start_frame"][-1] = (time()-self.times["finding_start_frame"][-1])
            
            
            self.up_and_down(image_paths,i,-1)
            self.empty_image_found = False
            self.up_and_down(image_paths,i,1)
            
            
            # Filtering the results for a folder and appending to the global results
            if self.res_cache.shape[0]<2:
                self.results.append(self.res_cache)
            else:
                self.results.append(self.filter_bboxes(self.res_cache))
            
             
            if self.save_results:
                self.times["total_time"].append(time() - st)
                
                self.mip_cache = cv2.imread(image_paths[-1])
                self.mip_cache = rectangles_on_mip(self.mip_cache, self.results[-1][:,:4].astype(int),box_format="xyxy_one", show_indices=True, color=(0,0,255))
                
                if path[-1]=='/':
                    path = path[:-1]
                    
                current_out_dir = os.path.join(self.out_path,f"{path.split('/')[-1]}_{datetime.now().strftime('%Y_%m_%d:%H_%M_%S')}")
                os.mkdir(current_out_dir)
                cv2.imwrite(os.path.join(current_out_dir,"detection.png"),self.mip_cache)
                
                with open(os.path.join(current_out_dir,"results.pkl"),"wb") as f:
                    pickle.dump((self.results[-1], self.times),f)
                    
            print(f"{self.color['green']}Detected {self.color['darkyellow']}{self.results[-1].shape[0]} {self.color['green']}Bacteria in {self.color['darkcyan']}{path}\n{self.color['green']}Detection time >> {self.color['darkyellow']}{(time()-st):.2f} sec.\n")
            
            if self.save_results:
                 print(f"{self.color['green']}Results saved in {self.color['darkcyan']}{current_out_dir}\n")
            print(f"{self.color['green']}{'-'*40}")
        
        print(f"{self.color['darkgreen']}\n\nDetection time for all folders {self.color['darkyellow']}{time()-start:.2f} sec.{self.color['white']}\n\n")

        return self.results
            
if __name__ == "__main__":
    # load custom plugin and engine
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="weights/model_square_fp16.engine", help='the path to the engine file')
    parser.add_argument('--source', type=str, help='path to the folder of images for detection')
    parser.add_argument('--frame-weights', type=str, default="weights/frame_weights.npy", help='path to the frame weights')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='The minimum confidence threshold for the detections')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='The IOU threshold for Yolo NMS')
    parser.add_argument('--agc-distance', type=int, default=15, help='The maximum distance between elements of cluster for bacteria')
    parser.add_argument('--out-path', type=str, default="./results", help='path to the frame weights')
    parser.add_argument('--save-results', action="store_true", help='path to the frame weights')
    
    
    args = parser.parse_args()

    PLUGIN_LIBRARY = "weights/libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    
    categories = ["Bacteria"]

    if not os.path.exists(args.source):
        raise NotADirectoryError(f"{args.source} is not a valid path")
    if not os.path.exists(args.engine):
        raise NotADirectoryError(f"{args.engine} is not a valid engine path")
    if not os.path.exists(args.frame_weights):
        raise NotADirectoryError(f"{args.frame_weights} is not a valid frame_weights path")
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        
    try:
        model = YoLov5TRT(engine_file_path=args.engine,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        out_path=args.out_path,
                        save_results=args.save_results,
                        agc_distance=args.agc_distance)
        predictions = model.infer(root_path=args.source, frame_weights=args.frame_weights) 
    finally:
        model.destroy()
