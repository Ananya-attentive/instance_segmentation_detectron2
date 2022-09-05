import os
import cv2
import warnings 
from utils import get_predictor, image_list, create_folder, get_mask_and_bbox_batching, save_image, get_combined_mask, remove_overlapping
warnings.filterwarnings("ignore")

                             
def inference_visualization(SaveDir, JsonPath, TifPath, predictor_remove, predictor_detect, Image_Size, buffer_percentage, mask_combine_threshold, mask_union_threshold,remove_threshold):

    create_folder(SaveDir)
    for image in image_list(JsonPath): 

        image = "18.tif"
        print(image)
  
        img = cv2.imread(os.path.join(TifPath, image))
        create_folder(os.path.join(SaveDir, "temp_folder_remove"))
        create_folder(os.path.join(SaveDir, "temp_folder_detect"))
        create_folder(os.path.join(SaveDir, "temp_folder_combine"))
        boxes_list_remove = get_mask_and_bbox_batching(img, Image_Size, predictor_remove, buffer_percentage, os.path.join(SaveDir, "temp_folder_remove"))
        boxes_list_detect = get_mask_and_bbox_batching(img, Image_Size, predictor_detect, buffer_percentage, os.path.join(SaveDir, "temp_folder_detect"))
        get_combined_mask(boxes_list_remove, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "temp_folder_remove"))
        get_combined_mask(boxes_list_detect, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "temp_folder_detect"))
        remove_overlapping(os.path.join(SaveDir, "temp_folder_detect"), os.path.join(SaveDir, "temp_folder_remove"), remove_threshold, os.path.join(SaveDir, "temp_folder_combine"))
        savePath  = os.path.join(SaveDir, image[:-4] + ".png")
        save_image(img, os.path.join(SaveDir, "temp_folder_combine"), savePath)
        break
       
    print("Images Saved in " + SaveDir)


if __name__ == "__main__":
      
    save_dir = "/home/workspace/DataVisualization/Bushes_data_batched_analysis"
    tif_path = "/home/workspace/bushes/batched_tif"
    image_size = 512
    json_path = "/home/workspace/bushes/bushes_train.json"
    weight_path_remove = "/home/workspace/detectron2/detectron2/output_tree_train/model_0056699.pth"
    weight_path_detect = "/home/workspace/detectron2/detectron2/output_combine/model_0022771.pth"
    config_path = "/home/workspace/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    buffer_percentage = 12.5
    threshold_roiHead = 0.7
    threshold_retinanet = 0.7
    mask_combine_threshold = 0.1
    mask_union_threshold = 0.95
    remove_threshold = 0.2

    predictor_remove = get_predictor(config_path, weight_path_remove, threshold_roiHead, threshold_retinanet)
    predictor_detect= get_predictor(config_path, weight_path_detect, threshold_roiHead, threshold_retinanet)

    inference_visualization(save_dir, json_path, tif_path,  predictor_remove, predictor_detect, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold)

   