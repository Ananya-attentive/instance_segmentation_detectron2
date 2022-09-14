import os
import cv2
import warnings 
from utils import create_folder, get_predictor, image_list, get_mask_and_bbox_batching, get_combined_mask, image_gt, get_iou, remove_overlapping
warnings.filterwarnings("ignore")

# detect = shrubs + trees
# 
                    
def result_analysis( SaveDir, JsonPath, TifPath, predictor_remove, predictor_detect, Image_Size, buffer_percentage, mask_combine_threshold, mask_union_threshold, iou_threshold):

    act_count = 0 
    pred_count = 0 
    mask_iou = 0 
    total_area = 0 

    create_folder(os.path.join(SaveDir, "trees"))
    create_folder(os.path.join(SaveDir, "vegetation"))
    create_folder(os.path.join(SaveDir, "shrubs"))


    for image in image_list(JsonPath): 
        image = "0.tif"
        
        
        act_box, act_class, act_mask = image_gt(JsonPath, image)
                
        act_count += len(act_class)
        img = cv2.imread(os.path.join(TifPath, image))
        
        boxes_list_trees = get_mask_and_bbox_batching(img, Image_Size, predictor_remove, buffer_percentage, os.path.join(SaveDir, "trees"))
        boxes_list_greenery = get_mask_and_bbox_batching(img, Image_Size, predictor_detect, buffer_percentage, os.path.join(SaveDir, "vegetation"))
       
        get_combined_mask(boxes_list_trees, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "trees"))
        get_combined_mask(boxes_list_greenery, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "vegetation"))
     

        remove_overlapping(os.path.join(SaveDir, "vegetation"), os.path.join(SaveDir, "trees"), remove_threshold, os.path.join(SaveDir, "shrubs"))
        
        correct_count, iou, area = get_iou(act_mask, os.path.join(SaveDir, "shrubs"), iou_threshold)
        
        pred_count += correct_count
        mask_iou += iou
        total_area += area
        break

    print("Accuracy is:      ")
    print(pred_count/act_count)
    print("Mask_iou is:       ")
    print(mask_iou/total_area)




if __name__ == "__main__":
      
   
    tif_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/shrubs_tif/batched_tif"
    save_dir = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/z1"
    image_size = 512
    json_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/shrubs_test.json"
    weight_path_remove = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/tree.pth"
    weight_path_detect = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/combine.pth"
    config_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    buffer_percentage = 12.5
    threshold_roiHead = 0.7
    threshold_retinanet = 0.7
    mask_combine_threshold = 0.1
    mask_union_threshold = 0.95
    iou_threshold = 0.2
    remove_threshold = 0.7


    predictor_remove = get_predictor(config_path, weight_path_remove, threshold_roiHead, threshold_retinanet)
    predictor_detect= get_predictor(config_path, weight_path_detect, threshold_roiHead, threshold_retinanet)
    
    result_analysis(save_dir, json_path, tif_path,  predictor_remove, predictor_detect, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, iou_threshold)

   