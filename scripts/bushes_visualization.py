import os
import cv2
import warnings 
import time
from utils import get_combined_mask, get_predictor, image_list, create_folder, get_mask_and_bbox_after_batching, save_image, get_combined_mask, remove_overlapping
warnings.filterwarnings("ignore")



# def inference_on_single_image():
                             
                             
def inference_visualization(output_dir, tifs_dir, json_path, predictor_vegetation, predictor_tree, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold):
    
    create_folder(output_dir)
    create_folder(os.path.join(output_dir, "vegetation"))
    create_folder(os.path.join(output_dir, "trees"))
    create_folder(os.path.join(output_dir, "bushes"))
   
    a = image_list(json_path)
    
       
    for image in image_list(json_path): 

        image = "18.tif"
        img = cv2.imread(os.path.join(tifs_dir, image))
        
        bbox_list_vegetation = get_mask_and_bbox_after_batching(img, image_size, predictor_vegetation, buffer_percentage, os.path.join(output_dir, "vegetation"))
        bbox_list_trees = get_mask_and_bbox_after_batching(img, image_size, predictor_tree, buffer_percentage, os.path.join(output_dir, "trees"))
        
        get_combined_mask(bbox_list_vegetation, mask_combine_threshold, mask_union_threshold, os.path.join(output_dir, "vegetation"))
        get_combined_mask(bbox_list_trees, mask_combine_threshold, mask_union_threshold, os.path.join(output_dir, "trees"))
        
        #bushes = vegetation-trees
        remove_overlapping(os.path.join(output_dir, "vegetation"), os.path.join(output_dir, "trees"), os.path.join(output_dir, "bushes"), remove_threshold)

        savePath  = os.path.join(output_dir, image.replace('.tif', '.png'))
        save_image(img, os.path.join(output_dir, "bushes"), savePath)
       
        break
    
    print("Images Saved in " + output_dir)
#6.25 gb RAM, 31 seconds(this one), 65 seconds(old one)


if __name__ == "__main__":
      
    tifs_dir = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/bushes_tif/batched_tif"
    save_dir = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/z1"
    image_size = 512
    json_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/bushes_test.json"
    config_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    weight_path_vegetation = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/vegetation.pth"
    weight_path_tree = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/tree.pth"
    
    
    buffer_percentage = 12.5
    threshold_roi_head = 0.7
    threshold_retinanet = 0.7
    mask_combine_threshold = 0.1
    mask_union_threshold = 0.95
    remove_threshold = 0.2

    predictor_vegetation = get_predictor(config_path, weight_path_vegetation, threshold_roi_head, threshold_retinanet)
    predictor_tree = get_predictor(config_path, weight_path_tree, threshold_roi_head, threshold_retinanet)
    

    inference_visualization(save_dir, tifs_dir, json_path, predictor_vegetation, predictor_tree, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold)

   