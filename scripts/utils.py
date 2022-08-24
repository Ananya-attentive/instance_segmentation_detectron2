from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import json
import cv2
from detectron2 import structures
import os
import shutil
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import torch 
from detectron2.structures.masks import polygons_to_bitmask


def register_data(JsonPath, TifPath, RegisterDataName):

    register_coco_instances(RegisterDataName, {}, JsonPath, TifPath)
    print("Data Registered")

def create_folder(SaveDir):
    
    if os.path.exists(SaveDir):
            shutil.rmtree(SaveDir)
    os.makedirs(SaveDir)


def get_predictor(config_path, Weight_path, threshold_roiHead, threshold_retinanet):

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = Weight_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold_retinanet
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_roiHead
    predictor = DefaultPredictor(cfg)

    return predictor

def image_list(JsonPath):

    image_list = []
    f = open(JsonPath,'r')
    image_data = json.loads(f.read())

    for image in image_data["images"]:
        image_list.append(image["file_name"])

    return image_list

def inference_image(im, PathDir, PartX, PartY, registerDataName, predictor):

    outputs = predictor(im)
    detectron2__metadata = MetadataCatalog.get(registerDataName)
    v = Visualizer(im[:, :, ::-1], metadata=detectron2__metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imwrite(os.path.join(PathDir, str(PartX) + "_" + str(PartY) +"_ML.png"),out.get_image()[:, :, ::-1])

    return out.get_image()[:, :, ::-1]



#def inference_image(im, Pathdir, registerDataName, predictor, image_size)



def inference_image_output(im,predictor):

    return predictor(im)

def iou_mask(mask1, mask2):
   
    intersection = torch.logical_and(mask2, mask1).to(torch.int).sum() 
    if intersection == 0:
        return 0.0 
    else:   
        union = torch.logical_or(mask2, mask1).to(torch.int).sum()
        return (intersection / union) 

def weighted_iou(mask1, mask2):

    area = len(np.where(mask1==True)[0])
   
    #print(mask1.shape, mask2.shape)
    intersection = torch.logical_and(torch.as_tensor(mask2).cuda(), torch.as_tensor(mask1).cuda()).to(torch.int).sum() #(mask2 * mask1).sum()
    if intersection == 0:
        return 0.0, area #, 0 #area
    else:
        union = torch.logical_or(torch.as_tensor(mask2).cuda(), torch.as_tensor(mask1).cuda()).to(torch.int).sum()
        return (intersection / union)*area , area



def check_union(mask, mask_union_threshold):

    intersection_count = torch.logical_and(mask[0], mask[1]).to(torch.int).sum() 
    union_count = torch.logical_or(mask[0], mask[1]).to(torch.int).sum()
    
    if intersection_count == 0:
        return False

    mask_union_cover_1 = len(list(zip(*np.where(mask[0].cpu() == True))))/union_count
    mask_union_cover_2 = len(list(zip(*np.where(mask[1].cpu() == True))))/union_count

    if mask_union_cover_1 > mask_union_threshold or mask_union_cover_2 > mask_union_threshold:
        return True
    else: 
        return False
    

def iou_calculation(act_bbox, pred_bbox, act_class,  pred_class, act_mask, pred_mask):
 
    IOUs = structures.pairwise_iou(act_bbox, pred_bbox)
    act = []
    pred = []
    bbox_iou = 0
    Area = 0 
    mask_iou = 0
    bbox_iou_perClass = [0]*len(max([act_class]))
    mask_iou_perClass = [0]*len(max([act_class]))
    
    for i in range(len(IOUs)):
        if len(IOUs[i])>0  :
            if len([j for j in list(IOUs[i]) if j != 0])>0:
                act.append(act_class[i])
                pred.append(int(pred_class[list(IOUs[i]).index(max(IOUs[i]))]+1))
                if act_class[i] == int(pred_class[list(IOUs[i]).index(max(IOUs[i]))]+1):
                    bbox_iou += max(IOUs[i])
                    bbox_iou_perClass[act_class[i]-1] += max(IOUs[i])
                    act_mask_bitMask = polygons_to_bitmask(act_mask[i], int(np.shape(pred_mask[list(IOUs[i]).index(max(IOUs[i]))])[0]), int(np.shape(pred_mask[list(IOUs[i]).index(max(IOUs[i]))])[1]))
                    mask_IOU, area = iou_mask(torch.as_tensor(act_mask_bitMask).cuda(), torch.as_tensor(pred_mask[list(IOUs[i]).index(max(IOUs[i]))]).cuda())
                    mask_iou += mask_IOU
                    mask_iou_perClass[act_class[i]-1] += mask_IOU
                    Area += area
    if Area == 0:
        mask_iou = 0
    #else:

    #    mask_iou = mask_iou/Area
    #print(mask_iou)
    
    return act, pred, bbox_iou, bbox_iou_perClass, mask_iou, mask_iou_perClass, Area

def image_gt(json_path, file_name):

    image_data = json.loads(open(json_path,'r').read())
    image_id = None
    for data in image_data["images"]:

        if data['file_name'] == file_name:
            image_id = data['id']
            break
    
    bbox_list = []
    class_list = []
    mask_list = []

    for i in range(len(image_data["annotations"])): 
        
        if image_data["annotations"][i]['image_id'] == image_id:
            bbox = image_data["annotations"][i]['bbox']

            bbox_list.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            class_list.append(image_data["annotations"][i]["category_id"])
            mask_list.append(image_data["annotations"][i]['segmentation'])    

    bbox_list = torch.as_tensor(bbox_list).cuda()
    bbox_list = structures.Boxes(bbox_list)

    return bbox_list, class_list, mask_list


def combine_overlapping_mask(mask_list, merged_pairs, bbox_iou_metric, merged_index, mask_combine_threshold, mask_union_threshold):

    mask = []
    merged_list = []
    idx_list = []

    for i in range(len(mask_list)):
        if i in idx_list:
            continue
        else:

            temp_mask = mask_list[i]

            for y in np.where(bbox_iou_metric[merged_pairs[i][0]].cpu()>0.1)[0]:
                if y in merged_index:
                    for u in range(len(merged_pairs)):
                        if (u,i) in merged_list:
                            continue
                        else:
                            if y in (merged_pairs[u][0], merged_pairs[u][1]):
                                if (y,i) != (merged_pairs[u][0], merged_pairs[u][1]) or (i,y) != (merged_pairs[u][0], merged_pairs[u][1]):
                                    iou = iou_mask(torch.as_tensor(temp_mask).cuda(), torch.as_tensor(mask_list[u]).cuda())
                                    if iou > mask_combine_threshold or check_union((torch.as_tensor(temp_mask).cuda(), torch.as_tensor(mask_list[u]).cuda()), mask_union_threshold):
                                        merged_list.append((i,u))
                                        temp_mask = (np.bitwise_or(temp_mask, mask_list[u]))
                                        idx_list.append(u)
                                        idx_list.append(i)

            for y in np.where(bbox_iou_metric[merged_pairs[i][1]].cpu()>0.1)[0]:
                if y in merged_index:
                    for u in range(len(merged_pairs)):
                        if (u,i) in merged_list:
                            continue
                        else:
                            if y in (merged_pairs[u][0], merged_pairs[u][1]):
                                if (y,i) != (merged_pairs[u][0], merged_pairs[u][1]) or (i,y) != (merged_pairs[u][0], merged_pairs[u][1]):
                                    iou = iou_mask(torch.as_tensor(temp_mask).cuda(), torch.as_tensor(mask_list[u]).cuda())
                                    if iou > mask_combine_threshold or check_union((torch.as_tensor(temp_mask).cuda(), torch.as_tensor(mask_list[u]).cuda()), mask_union_threshold):
                                        merged_list.append((i,u))
                                        temp_mask = (np.bitwise_or(temp_mask,mask_list[u]))
                                        idx_list.append(u)
                                        idx_list.append(i)
            mask.append(temp_mask)

   


    return mask
 

def get_combined_mask(mask_list, boxes_list, mask_combine_threshold, mask_union_threshold):

    structured_boxes_list= structures.Boxes(torch.as_tensor(boxes_list).cuda())
    bbox_iou_metric = structures.pairwise_iou(structured_boxes_list, structured_boxes_list)

    merged_pairs = []
    temp_mask_list = []
    merged_index = []

    for i in range(len(bbox_iou_metric)):

        idx_list = np.where(bbox_iou_metric[i].cpu()>0)[0]
        if len(idx_list)>1:
            for idx in idx_list:
                if i != idx:
                    if (i, idx) in merged_pairs:
                        continue
                    else:
                        iou = iou_mask(torch.as_tensor(mask_list[idx]).cuda(), torch.as_tensor(mask_list[i]).cuda())
                        if iou > mask_combine_threshold or check_union((torch.as_tensor(mask_list[idx]).cuda(), torch.as_tensor(mask_list[i]).cuda()), mask_union_threshold):
                            merged_pairs.append((idx, i))
                            temp_mask_list.append(np.bitwise_or(mask_list[i],mask_list[idx] ))
                            merged_index.append(i)
                            merged_index.append(idx)
                        


    temp_mask_list = combine_overlapping_mask(temp_mask_list, merged_pairs, bbox_iou_metric, merged_index, mask_combine_threshold, mask_union_threshold)

    for i in range(len(mask_list)):
        if i in merged_index:
            continue
        else:
            temp_mask_list.append(mask_list[i])
    return temp_mask_list


def get_mask_and_bbox_batching(img, image_size, predictor, buffer_percentage, folder_path):

    create_folder(os.path.join(folder_path, "mask_original"))
    create_folder(os.path.join(folder_path, "bbox_original"))

    count_boxes = 1 
    count_mask = 1

    imgwidth=img.shape[0]
    imgheight=img.shape[1]

    split_horizontal = 1 if round(imgwidth/image_size) == 0 else round(imgwidth/image_size)
    split_verticle = 1 if round(imgheight/image_size) == 0 else round(imgheight/image_size)

    batched_image_width = imgwidth//split_horizontal
    batched_image_heigth = imgheight//split_verticle

    mask_list = []
    boxes_list = []

    for i in range(split_verticle):
        for j in range(split_horizontal):

            buffer_min_x = 0 if i == 0 else batched_image_heigth/int(100/buffer_percentage)
            buffer_min_y = 0 if j == 0 else batched_image_width/int(100/buffer_percentage)
            buffer_max_y = batched_image_width/int(100/buffer_percentage)
            buffer_max_x = batched_image_heigth/int(100/buffer_percentage)
        
            min_bound_x = int(i*batched_image_heigth - buffer_min_x)
            min_bound_y = int(j*batched_image_width- buffer_min_y)
            max_bound_x = imgheight if split_verticle - 1 == i else int(batched_image_heigth*(i+1) + buffer_max_x)
            max_bound_y = imgwidth if split_horizontal - 1 == j else int(batched_image_width*(j+1) + buffer_max_y)

            batched_image = img[min_bound_y: max_bound_y, min_bound_x:max_bound_x]
            batched_prediction = predictor(batched_image)
            masks = batched_prediction["instances"].pred_masks
            bboxs = batched_prediction["instances"].pred_boxes
            
            for box in bboxs:
                boxes_list.append([box[0]+ min_bound_x, box[1]+ min_bound_y, box[2]+ min_bound_x, box[3]+ min_bound_y])
                with open(os.path.join(folder_path, "bbox_original", str(count_boxes) + ".txt"), 'w') as f:
                    for box_point in box[0]+ min_bound_x, box[1]+ min_bound_y, box[2]+ min_bound_x, box[3]+ min_bound_y:
                        f.write(str(box_point)+'\n')
                    f.close()

                count_boxes += 1
                    

            for mask in masks:
                temp_mask = np.full((imgwidth, imgheight), False,  dtype=bool)
                for true_point in list(zip(*np.where(np.array(mask.cpu()) == True))) :   
                    temp_mask[true_point[0] + min_bound_y][true_point[1] + min_bound_x]= True               
                mask_list.append(temp_mask)
                with open(os.path.join(folder_path, "mask_original", str(count_mask) + ".txt"), 'w') as f:
                    for mask_point in temp_mask:
                        f.write(str(mask_point)+'\n')
                    f.close()
                count_mask += 1
            #count += 1
            
    
    return mask_list, boxes_list

    
def save_image(img, masks, savePath):

    increment = 0
    color = 30

    for mask in masks:
   
        img[np.array(mask)*125 == 125] = [245, 0, 0]#[color*increment, color*increment, color*increment]
        increment += 1
        if color*increment > 255 :
            increment = 0 
                        
    cv2.imwrite(savePath, img)

def check_overlapping(mask, mask_overlapping, remove_threshold):

    for mask_overlap in mask_overlapping:
    
        if iou_mask(torch.as_tensor(mask).cuda(), torch.as_tensor(mask_overlap).cuda()) > remove_threshold:
            return False
    return True

def remove_overlapping(mask_detect, mask_remove, remove_threshold):

    removed_mask = []
    for mask in mask_detect:
        if check_overlapping(mask, mask_remove, remove_threshold):
            removed_mask.append(mask)

    return removed_mask


def get_iou(act_mask_list, pred_mask_list, iou_threshold):

    mask_iou = 0
    total_area = 0
    pred_count = 0
    for act_mask in act_mask_list:
        iou_list = []
        area_list = []
        for pred_mask in pred_mask_list:
            act_mask_bitMask = polygons_to_bitmask(act_mask, int(pred_mask.shape[0]), int(pred_mask.shape[1]))
            iou, area = weighted_iou(act_mask_bitMask, pred_mask)
            if iou/area > iou_threshold:
                iou_list.append(iou/area)
                area_list.append(area)


        if len(iou_list)> 0 : 
            mask_iou += max(iou_list)*area_list[iou_list.index(max(iou_list))]
            total_area += area_list[iou_list.index(max(iou_list))]
            if max(iou_list) > 0:
                pred_count += 1

    
    return  pred_count, mask_iou, total_area




    
    
  
 


    