import os
import cv2
import warnings 
from utils import get_predictor, image_list, create_folder, get_mask_and_bbox_after_batching, save_image, get_combined_mask, remove_overlapping
warnings.filterwarnings("ignore")

                             
def inference_visualization(SaveDir, JsonPath, TifPath, predictor_remove, predictor_detect, Image_Size, buffer_percentage, mask_combine_threshold, mask_union_threshold,remove_threshold):

    create_folder(SaveDir)
    create_folder(os.path.join(SaveDir, "tmp_greenery"))
    create_folder(os.path.join(SaveDir, "tmp_trees"))
    create_folder(os.path.join(SaveDir, "tmp_combined"))
    
    
    for image in image_list(JsonPath): 

        image = "18.tif"
        img = cv2.imread(os.path.join(TifPath, image))
        
        
        boxes_list_trees = get_mask_and_bbox_after_batching(img, Image_Size, predictor_remove, buffer_percentage, os.path.join(SaveDir, "tmp_trees"))
        boxes_list_greenery = get_mask_and_bbox_after_batching(img, Image_Size, predictor_detect, buffer_percentage, os.path.join(SaveDir, "tmp_greenery"))
        
        
        import time
        t1 = time.time()
        get_combined_mask(boxes_list_trees, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "tmp_trees"))
        t2 = time.time()
        print(t2-t1)
        
        t3 = time.time()
        get_combined_mask(boxes_list_greenery, mask_combine_threshold, mask_union_threshold, os.path.join(SaveDir, "tmp_greenery"))
        t4 = time.time()
        print(t4-t3)
        
        remove_overlapping(os.path.join(SaveDir, "tmp_greenery"), os.path.join(SaveDir, "tmp_trees"), remove_threshold, os.path.join(SaveDir, "tmp_combined"))
        t6 = time.time()
        print(t6-t4)
        
        
        t7 = time.time()
        print(t7-t6)
        
        savePath  = os.path.join(SaveDir, image.replace('.tif', '.png'))
        save_image(img, os.path.join(SaveDir, "tmp_combined"), savePath)
        break
       
    print("Images Saved in " + SaveDir)


if __name__ == "__main__":
      
    tif_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/bushes_tif/batched_tif"
    save_dir = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/z1"
    image_size = 512
    json_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/bushes_test.json"
    weight_path_remove = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/tree.pth"
    weight_path_detect = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/combine.pth"
    config_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    buffer_percentage = 12.5
    threshold_roiHead = 0.7
    threshold_retinanet = 0.7
    mask_combine_threshold = 0.1
    mask_union_threshold = 0.95
    remove_threshold = 0.2

    predictor_remove = get_predictor(config_path, weight_path_remove, threshold_roiHead, threshold_retinanet)
    predictor_detect= get_predictor(config_path, weight_path_detect, threshold_roiHead, threshold_retinanet)

    inference_visualization(save_dir, json_path, tif_path,  predictor_remove, predictor_detect, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold)

   