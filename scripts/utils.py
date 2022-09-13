from xml.dom.expatbuilder import InternalSubsetExtractor
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
import shapely
import torch 
import geopandas as gpd
from shapely.geometry import Polygon, mapping

from detectron2.structures.masks import polygons_to_bitmask

import sys
from qgis.core import QgsApplication
from qgis.analysis import QgsNativeAlgorithms
QgsApplication.setPrefixPath('/usr', True) #for avoiding "Application path not initialized"
sys.path.append('/usr/share/qgis/python/plugins')
import processing
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())




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
    return out.get_image()[:, :, ::-1]


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
    intersection = torch.logical_and(torch.as_tensor(mask2).cuda(), torch.as_tensor(mask1).cuda()).to(torch.int).sum() #(mask2 * mask1).sum()
    if intersection == 0:
        return 0.0, area #, 0 #area
    else:
        union = torch.logical_or(torch.as_tensor(mask2).cuda(), torch.as_tensor(mask1).cuda()).to(torch.int).sum()
        return (intersection / union)*area , area



def check_union(mask, mask_union_threshold):

    intersection_count = mask[0].intersection(mask[1]).area
    union_count = mask[0].union(mask[1]).area
    intersection_thresh = 0.9

    if intersection_count == 0:
        return False

    mask_union_cover_1 = mask[0].area/union_count
    mask_union_cover_2 = mask[1].area/union_count
    intersected_part1 = mask[0].area/intersection_count
    intersected_part2 = mask[1].area/intersection_count

    if mask_union_cover_1 > mask_union_threshold or mask_union_cover_2 > mask_union_threshold or intersected_part1 > intersection_thresh or intersected_part2 > intersection_thresh:
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


def combine_overlapping_mask(merged_pairs, bbox_iou_metric, merged_index, mask_combine_threshold, mask_union_threshold, folder_path):

    merged_list = []
    idx_list = []
    final_count = 0 
    
    mask_gdf = gpd.read_file(os.path.join(folder_path, "mask_combined_1.geojson"))
    
    geometries = []
    mask_ids = []
    
    for i in range(mask_gdf.shape[0]):
        if i in idx_list:
            continue
        else:
            temp_mask = mask_gdf['geometry'].iloc[i]
            
            
            for y in np.where(bbox_iou_metric[merged_pairs[i][0]].cpu()>0.1)[0]:
                if y in merged_index:
                    for u in range(len(merged_pairs)):
                        if (u,i) in merged_list:
                            continue
                        else:
                            if y in (merged_pairs[u][0], merged_pairs[u][1]):
                                if (y,i) != (merged_pairs[u][0], merged_pairs[u][1]) or (i,y) != (merged_pairs[u][0], merged_pairs[u][1]):
                                    mask = mask_gdf['geometry'].iloc[u]
                                    
                                    iou = temp_mask.intersection(mask).area/temp_mask.union(mask).area
                                    if iou > mask_combine_threshold or check_union((temp_mask, mask), mask_union_threshold):
                                        merged_list.append((i,u))
                                        temp_mask = temp_mask.union(mask)            
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
                                    mask = mask_gdf['geometry'].iloc[u]
                                    iou = temp_mask.intersection(mask).area/temp_mask.union(mask).area
                                    if iou > mask_combine_threshold or check_union((temp_mask, mask), mask_union_threshold):
                                        merged_list.append((i,u))
                                        temp_mask = temp_mask.union(mask) 
                                        idx_list.append(u)
                                        idx_list.append(i)
     
     
            geometries.append(temp_mask)
            mask_ids.append(final_count)
            final_count += 1
            
    grid = gpd.GeoDataFrame()
    grid['geometry'] = geometries
    grid['mask_id'] = mask_ids
    grid.to_file(os.path.join(folder_path, "mask_combined_2.geojson"), driver='GeoJSON')  
    return grid 
            

# def get_combined_mask(boxes_list, mask_combine_threshold, mask_union_threshold, folder_path):
    
#     #combining overlapping mask
#     structured_boxes_list= structures.Boxes(torch.as_tensor(boxes_list).cuda())
#     bbox_iou_metric = structures.pairwise_iou(structured_boxes_list, structured_boxes_list)
            
#     merged_pairs = []
#     merged_index = []
#     merged_count = 0 

#     mask_df = gpd.read_file(os.path.join(folder_path, "mask_original.geojson"))

#     geometries = []
#     mask_ids = []


#     for i in range(len(bbox_iou_metric)):
                
#         mask_1 = mask_df['geometry'].iloc[i].buffer(0)
#         mask_id_1 = mask_df['mask_id'].iloc[i]

#         idx_list = np.where(bbox_iou_metric[i].cpu()>0)[0]  
        
#         # print(i, idx_list)
#         # continue
        
#         if len(idx_list)==1:
#             geometries.append(mask_1)
#             mask_ids.append(f'{mask_id_1}')
            
        
#         elif len(idx_list)>1:
#             mask_union_geometry = mask_1
#             mask_union_id = f'{mask_id_1}'
#             for idx in idx_list:
#                 if i != idx:
#                     if (i, idx) in merged_pairs:
#                         continue
#                     else:
#                         mask_2 = mask_df['geometry'].iloc[idx].buffer(0)
#                         mask_id_2 = mask_df['mask_id'].iloc[idx]

#                         iou = (mask_1.intersection(mask_2).area)/(mask_1.union(mask_2).area)
                        
                        
#                         if iou > mask_combine_threshold or check_union((mask_1, mask_2), mask_union_threshold):
#                             mask_union_geometry = mask_union_geometry.union(mask_2)
#                             # merged_pairs.append((idx, i))
#                             # geometries.append(mask_1.union(mask_2))
#                             mask_union_id += f'_{mask_id_2}'
                            
#                             # merged_count += 1 
#                             # merged_index.append(i)
#                             # merged_index.append(idx)
                        
#                         # else:
#                         #     geometries.append(mask_2)
#                         #     mask_ids.append(f'{mask_id_2}')
#             geometries.append(mask_union_geometry)
#             mask_ids.append(mask_union_id)
#     merged_mask_gdf = gpd.GeoDataFrame()
#     merged_mask_gdf['geometry'] = geometries  
#     merged_mask_gdf['mask_id'] = mask_ids     
#     merged_mask_gdf.to_file(os.path.join(folder_path, "mask_combined_1.geojson"), driver='GeoJSON')
#     # exit()
    
#     #again combining overlapping mask
#     final_mask = combine_overlapping_mask(merged_pairs, bbox_iou_metric, merged_index, mask_combine_threshold, mask_union_threshold, folder_path)
#     final_mask_count = final_mask.shape[0]   

#     mask_original_gdf = gpd.read_file(os.path.join(folder_path, "mask_original.geojson"))
  
#     geometries = []
#     mask_ids = []
  
#     for i in range(mask_original_gdf.shape[0]):
#         if i in merged_index:
#             continue
#         else:
#             geometry = list(mask_original_gdf[mask_original_gdf['mask_id']==i]['geometry'])[0]
#             geometries.append(geometry)
#             mask_ids.append(final_mask_count)
#             final_mask_count += 1
    
#     final_mask_gdf = gpd.GeoDataFrame()
#     final_mask_gdf['geometry'] = geometries
#     final_mask_gdf['mask_id'] = mask_ids
#     final_mask_gdf.to_file(os.path.join(folder_path, "mask_combined_3.geojson"), driver='GeoJSON')
#     return final_mask_gdf
    

def iou(g1, g2):
    i = g1.intersection(g2).area
    u = g1.union(g2).area
    
    if u==0:
        return 0
    
    return i/u



def combine_overlapping_masks(mask_gdf, iou_threshold):
    
    mask_geometries = list(mask_gdf['geometry'])
    
    geometries = []
    visited_geoms = {}
    
    for i in range(mask_gdf.shape[0]):
        visited_geoms[i] = False
    
    i = 0
    
    for i in range(len(mask_geometries)):
        current_mask = mask_geometries[i]            
        visited_geoms[i] = True
        
        for j in range(i+1, len(mask_geometries)):
            if visited_geoms[i] == True:
                continue
            
            tmp_mask = mask_geometries[j]
            
            if iou(current_mask, tmp_mask)>iou_threshold:
                current_mask = current_mask.union(tmp_mask)
                visited_geoms[j] = True
                

        geometries.append(current_mask)
    
    combined_mask_gdf = gpd.GeoDataFrame()
    combined_mask_gdf['geometry'] = geometries  
    combined_mask_gdf['mask_id'] = [i for i in range(combined_mask_gdf.shape[0])]    

    return combined_mask_gdf


def get_combined_mask(boxes_list, mask_combine_threshold, mask_union_threshold, output_dir, layer):
   

   
    mask_original = gpd.read_file(os.path.join(output_dir, f'{layer}_original.geojson'))
    
    combined_mask_1 = combine_overlapping_masks(mask_original, mask_union_threshold)
    combined_mask_1.to_file(os.path.join(output_dir, f'{layer}_combined_1.geojson'), driver='GeoJSON')
    
    combined_mask_2 = combine_overlapping_masks(combined_mask_1, mask_union_threshold)
    combined_mask_2.to_file(os.path.join(output_dir, f'{layer}_combined_2.geojson'), driver='GeoJSON')
    
    

def batch_images(img, buffer_percentage):

    imgwidth=img.shape[0]
    imgheight=img.shape[1]

    split_horizontal = 10
    split_verticle = 10

    batched_image_width = imgwidth//split_horizontal
    batched_image_heigth = imgheight//split_verticle

    images = []

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
            images.append([batched_image, [min_bound_y, min_bound_x], [i,j]])
    
    return images


def swap_xy(input_path, output_path):
    processing.run("native:swapxy", {'INPUT':input_path,'OUTPUT':output_path})
   
def rotate(input_path, output_path, angle):   
    processing.run("native:affinetransform", {'INPUT':input_path,'DELTA_X':0,'DELTA_Y':0,'DELTA_Z':0,'DELTA_M':0,'SCALE_X':1,'SCALE_Y':1,'SCALE_Z':1,'SCALE_M':1,'ROTATION_Z':angle,'OUTPUT':output_path})


def get_mask_and_bbox_after_batching(img, image_size, predictor, buffer_percentage, output_dir, layer):

    imgwidth=img.shape[0]
    imgheight=img.shape[1]

    split_horizontal = 1 if round(imgwidth/image_size) == 0 else round(imgwidth/image_size)
    split_verticle = 1 if round(imgheight/image_size) == 0 else round(imgheight/image_size)

    batched_image_width = imgwidth//split_horizontal
    batched_image_heigth = imgheight//split_verticle

    boxes_list = []
    geometries = []    

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
            
            for mask in masks:
                temp_mask = np.full((imgwidth, imgheight), 0,  dtype=int)
                for true_point in list(zip(*np.where(np.array(mask.cpu()) == True))) :   
                    temp_mask[true_point[0] + min_bound_y][true_point[1] + min_bound_x]= 254

                temp_mask = temp_mask.astype(np.uint8)
                ret, thresh = cv2.threshold(temp_mask, 240, 255, cv2.THRESH_BINARY)
                cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(c) for c in cnts]
                max_index = np.argmax(areas)
                cnt=cnts[max_index]

                polygon = []
                for points in cnt:
                    polygon.append((points[0][0], points[0][1]))
            
                geometry = Polygon(polygon).buffer(0)
                geometries.append(geometry)
                
    
    
    #incorrect alignment due to polygonization of masks
    mask_unaligned_gdf = gpd.GeoDataFrame()
    mask_unaligned_gdf['geometry'] = geometries
    mask_unaligned_gdf['mask_id'] = [i for i in range(mask_unaligned_gdf.shape[0])]
    mask_unaligned_gdf.to_file(os.path.join(output_dir, f'{layer}_original_non_aligned.geojson'), driver='GeoJSON')
    
    #horizontally flipped polygons to correct the alignment
    mask_original_gdf = gpd.GeoDataFrame()
    mask_original_gdf['geometry'] = mask_unaligned_gdf['geometry'].apply(lambda x: shapely.affinity.scale(x, yfact = -1, origin = (1, 0)))
    mask_original_gdf['mask_id'] = [i for i in range(mask_original_gdf.shape[0])]
    mask_original_gdf.to_file(os.path.join(output_dir, f'{layer}_original.geojson'), driver='GeoJSON')
    
    return boxes_list

    
    
def save_image(img, folder_path, savePath):

    increment = 0
    color = 30

    shp_file = [x for x in os.listdir(folder_path) if x.endswith(".geojson")]

    for file_name in shp_file:

        mask = gpd.read_file(os.path.join(folder_path, file_name))['geometry'][0]

        if type(mask) == shapely.geometry.multipolygon.MultiPolygon:
            for poly in mask:

                points = []
                for pnts in mapping(poly)['coordinates'][0]:
                        points.append((int(pnts[0]),int(pnts[1])))
                
                points = np.array(points)
                points = points.reshape((-1, 1, 2))
                img = cv2.fillPoly(img, [points],[color*increment, color*increment, color*increment]) 
     
               
        elif type(mask) == shapely.geometry.polygon.Polygon:
           
            points = []
            for pnts in mapping(mask)['coordinates'][0]:
                    points.append((int(pnts[0]),int(pnts[1])))
            
            points = np.array(points)
            points = points.reshape((-1, 1, 2))
            img = cv2.fillPoly(img, [points],[color*increment, color*increment, color*increment]) 
     
        increment += 1
        if color*increment > 255 :
            increment = 0 
                        
    cv2.imwrite(savePath, img)
  

def check_overlap(vegetation_mask, trees_mask_list, iou_threshold):

    for tree_mask in trees_mask_list:
        i = vegetation_mask.intersection(tree_mask).area
        u = vegetation_mask.union(tree_mask).area
        
        if  u!=0 and i/u > iou_threshold :
            return True
        
    return False



def remove_overlapping(vegetation_path, trees_path, bushes_path, remove_threshold):

    vegetation_gdf = gpd.read_file(vegetation_path)
    vegetation_masks = list(vegetation_gdf['geometry'])
    
    trees_gdf = gpd.read_file(trees_path)
    trees_masks = list(trees_gdf['geometry'])
    
    geometries = []
    
    for vegetation_mask in vegetation_masks:        
        if check_overlap(vegetation_mask, trees_masks, remove_threshold) == False:
            geometries.append(vegetation_mask)
            
    mask_final = gpd.GeoDataFrame()
    mask_final['geometry'] = geometries
    mask_final['mask_id']=  [i for i in range(mask_final.shape[0])]
    mask_final.to_file(bushes_path, driver='GeoJSON')
    return mask_final


def get_iou(act_mask_list, pred_mask_path, iou_threshold):

    mask_iou = 0
    total_area = 0
    pred_count = 0
        
    
    for act_mask in act_mask_list:
        iou_list = []
        area_list = []
        for pred_mask_file in os.listdir(pred_mask_path):
            
            pred_mask = np.load(os.path.join(pred_mask_path, pred_mask_file))
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




    
    
  
 


    