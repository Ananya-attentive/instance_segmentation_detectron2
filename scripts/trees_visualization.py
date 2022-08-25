import os
import cv2
import warnings 
from utils import get_predictor, image_list, create_folder, get_mask_and_bbox_batching, save_image, get_combined_mask
warnings.filterwarnings("ignore")

                             
def inference_visualization(SaveDir, JsonPath, TifPath, predictor, Image_Size, buffer_percentage, mask_combine_threshold, mask_union_threshold):

    create_folder(SaveDir)

    for image in image_list(JsonPath): 

        image = "1.tif"

  
        img = cv2.imread(os.path.join(TifPath, image))
        #img = cv2.imread("/home/workspace/production_data/6.tif")
        create_folder(os.path.join(SaveDir, "temp_folder"))
        boxes_list = get_mask_and_bbox_batching(img, Image_Size, predictor, buffer_percentage, os.path.join(SaveDir, "temp_folder"))
        get_combined_mask(boxes_list, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "temp_folder"))

        save_image(img, os.path.join(os.path.join(SaveDir, "temp_folder"),"final_mask"), os.path.join(SaveDir, image[:-4] + ".png"))
        break
       
    print("Images Saved in " + SaveDir)


if __name__ == "__main__":
      
    save_dir = "/home/workspace/DataVisualization/Tree_data_batched_analysis"
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

    predictor = get_predictor(config_path, weight_path, threshold_roiHead, threshold_retinanet)
    inference_visualization(save_dir, json_path, tif_path,  predictor, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold)

   