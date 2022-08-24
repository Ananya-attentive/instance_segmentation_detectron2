import os
import cv2
import warnings 
from utils import get_predictor, image_list, create_folder, get_mask_and_bbox_batching, get_combined_mask, image_gt, get_iou
warnings.filterwarnings("ignore")

                             
def result_analysis(JsonPath, TifPath, predictor, Image_Size, buffer_percentage, mask_combine_threshold, mask_union_threshold, iou_threshold):

  
    act_count = 0 
    pred_count = 0 
    mask_iou = 0 
    total_area = 0 

    for image in image_list(JsonPath): 

    

        act_box, act_class, act_mask = image_gt(JsonPath, image)
        act_count += len(act_class)
        img = cv2.imread(os.path.join(TifPath, image))
        mask_list, boxes_list = get_mask_and_bbox_batching(img, Image_Size, predictor, buffer_percentage)
        mask = get_combined_mask(mask_list, boxes_list, mask_combine_threshold, mask_union_threshold)




        correct_count, iou, area = get_iou(act_mask, mask, iou_threshold)
        pred_count += correct_count
        mask_iou += iou
        total_area += area

    print("Accuracy is:      ")
    print(pred_count/act_count)

    print("Mask_iou is:       ")
    print(mask_iou/total_area)




if __name__ == "__main__":
      
  
    tif_path = "/home/workspace/nonBatchTree/batched_tif"
    image_size = 512
    json_path = "/home/workspace/nonBatchTree/tree_nonBatchtrain.json"
    weight_path = "/home/workspace/detectron2/detectron2/output_tree_train/model_0056699.pth"
    config_path = "/home/workspace/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    buffer_percentage = 12.5
    threshold_roiHead = 0.7
    threshold_retinanet = 0.7
    mask_combine_threshold = 0.1
    mask_union_threshold = 0.95
    iou_threshold = 0.2

    predictor = get_predictor(config_path, weight_path, threshold_roiHead, threshold_retinanet)
    result_analysis(json_path, tif_path,  predictor, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, iou_threshold)

   