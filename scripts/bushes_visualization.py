import os
import cv2
import yaml
from utils import get_combined_mask, get_predictor, image_list, create_folder, get_mask_and_bbox_after_batching, save_image, get_combined_mask, remove_overlapping
import warnings 
warnings.filterwarnings('ignore')


 

                  
def inference(tif_path, config):
    
 
    model_config = config['model_config']
    output_dir = config['output_dir']

    checkpoint_vegetation = config['checkpoint_vegetation']
    checkpoint_tree = config['checkpoint_tree']
    
    image_size = config['image_size']
    buffer_percentage = config['buffer_percentage']
    threshold_roi_head = config['threshold_roi_head']
    threshold_retinanet = config['threshold_retinanet']
    mask_combine_threshold = config['mask_combine_threshold']
    mask_union_threshold = config['mask_union_threshold']
    remove_threshold = config['remove_threshold']


    if not os.path.exists(checkpoint_vegetation):
        print(f"vegetation checkpoint doesn't exist at {checkpoint_vegetation}")
        return
    if not os.path.exists(checkpoint_tree):
        print(f"tree checkpoint doesn't exist at {checkpoint_tree}")
        return
    
        
    create_folder(output_dir)

    img = cv2.imread(tif_path)
    
    predictor_vegetation = get_predictor(model_config, checkpoint_vegetation, threshold_roi_head, threshold_retinanet)
    predictor_tree = get_predictor(model_config, checkpoint_tree, threshold_roi_head, threshold_retinanet)
    
    bbox_list_vegetation = get_mask_and_bbox_after_batching(img, image_size, predictor_vegetation, buffer_percentage, output_dir, 'vegetation')
    bbox_list_trees = get_mask_and_bbox_after_batching(img, image_size, predictor_tree, buffer_percentage, output_dir, 'trees')
    
    get_combined_mask(bbox_list_vegetation, mask_combine_threshold, mask_union_threshold, output_dir, 'vegetation')
    get_combined_mask(bbox_list_trees, mask_combine_threshold, mask_union_threshold, output_dir, 'trees')
    
    # bushes = vegetation-trees
    remove_overlapping(os.path.join(output_dir, 'vegetation_combined_2.geojson'), 
                        os.path.join(output_dir, 'trees_combined_2.geojson'), 
                        os.path.join(output_dir, 'bushes.geojson'), 
                        remove_threshold)
   
    
    
    
#6.25 gb RAM, 31 seconds(this one), 65 seconds(old one)


if __name__ == '__main__':
      
    config = yaml.safe_load(open('../shrubs_trees_config.yaml','r'))
    inference('/home/amit/Desktop/task_76_shrubs_integration/instance_segmentation_detectron2_new/bushes_tif/batched_tif/18.tif', config)
   