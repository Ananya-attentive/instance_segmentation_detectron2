import os
import cv2
import warnings 
import time
from utils import get_predictor, image_list, create_folder, get_mask_and_bbox_after_batching, save_image, get_combined_mask, remove_overlapping
warnings.filterwarnings("ignore")

                             
def inference_visualization(output_dir, json_path, tifs_path,  predictor_vegetation, predictor_tree, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold):
    
    create_folder(output_dir)
    create_folder(os.path.join(output_dir, "vegetation"))
    create_folder(os.path.join(output_dir, "trees"))
    create_folder(os.path.join(output_dir, "bushes"))
   
    
    for image in image_list(json_path): 

        image = "18.tif"
        img = cv2.imread(os.path.join(tifs_path, image))
        
        
        t1 = time.time()
        boxes_list_trees = get_mask_and_bbox_after_batching(img, image_size, predictor_tree, buffer_percentage, os.path.join(output_dir, "trees"))
        t2 = time.time()
        print(t2-t1, 'get mask and bbox trees')
        
        boxes_list_greenery = get_mask_and_bbox_after_batching(img, image_size, predictor_vegetation, buffer_percentage, os.path.join(output_dir, "vegetation"))
        t3 = time.time()
        print(t3-t2, 'get mask and bbox greenery')
        
        
        get_combined_mask(boxes_list_trees, mask_combine_threshold, mask_union_threshold, os.path.join(output_dir, "trees"))
        t4 = time.time()
        print(t4-t3, 'get combined_mask trees')
        
        
        get_combined_mask(boxes_list_greenery, mask_combine_threshold, mask_union_threshold, os.path.join(output_dir, "vegetation"))
        t5 = time.time()
        print(t5-t4, 'get combined mask greenery')
        
        remove_overlapping(os.path.join(output_dir, "vegetation"), os.path.join(output_dir, "trees"), remove_threshold, os.path.join(output_dir, "bushes"))
        t6 = time.time()
        print(t6-t5, 'remove overlapping')
        
        
        savePath  = os.path.join(output_dir, image.replace('.tif', '.png'))
        save_image(img, os.path.join(output_dir, "bushes"), savePath)
        t7 = time.time()
        print(t7-t6, 'save png')
        
        print(t7-t1, 'total time')
        
        break
    
    print("Images Saved in " + output_dir)
#6.25 gb RAM, 31 seconds(this one), 65 seconds(old one)


if __name__ == "__main__":
      
    tifs_path = "/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2/bushes_tif/batched_tif"
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
    

    inference_visualization(save_dir, json_path, tifs_path, predictor_vegetation, predictor_tree, image_size, buffer_percentage, mask_combine_threshold, mask_union_threshold, remove_threshold)

   