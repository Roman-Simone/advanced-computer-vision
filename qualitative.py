from xfeat_wrapper_copy import XFeatWrapper
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def visualize_comparisons(image1, image2, p1, p2, original_p1, original_p2):
    """
    Visualize two sets of correspondences (p1-p2 and original_p1-original_p2) 
    between two images in a stacked layout.
    
    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).
        p1: List of points in the first image (from your method).
        p2: List of corresponding points in the second image (from your method).
        original_p1: List of original points in the first image (from xfeat's method).
        original_p2: List of original corresponding points in the second image.
    """
    def create_combined_canvas(image1, image2, points1, points2, color_points=(0, 255, 0), color_lines=(255, 0, 0)):
        """
        Helper function to create a combined canvas with correspondences.
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        height = max(h1, h2)
        canvas1 = np.zeros((height, w1, 3), dtype=np.uint8)
        canvas2 = np.zeros((height, w2, 3), dtype=np.uint8)

        canvas1[:h1, :w1, :] = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        canvas2[:h2, :w2, :] = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        combined_image = np.hstack((canvas1, canvas2))
        offset_x = w1  # Offset for points in the second image

        for (x1, y1), (x2, y2) in zip(points1, points2):
            # Draw points
            cv2.circle(combined_image, (int(x1), int(y1)), 5, color_points, -1)  # Points in image1
            cv2.circle(combined_image, (int(x2) + offset_x, int(y2)), 5, color_points, -1)  # Points in image2
            # Draw connecting lines
            cv2.line(combined_image, (int(x1), int(y1)), (int(x2) + offset_x, int(y2)), color_lines, 2)

        return combined_image

    # Create two combined canvases: one for the inferred points, one for the original points
    canvas1 = create_combined_canvas(image1, image2, p1, p2, color_points=(0, 255, 0), color_lines=(255, 0, 0))
    canvas2 = create_combined_canvas(image1, image2, original_p1, original_p2, color_points=(0, 0, 255), color_lines=(0, 255, 255))

    # Stack the two canvases vertically
    stacked_canvas = np.vstack((canvas1, canvas2))

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(stacked_canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Comparison of Correspondences")
    plt.show()


def visualize_correspondences(image1, image2, p1, p2):
    """
    Visualize two images side by side with corresponding points linked by segments.
    
    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).
        p1: List of points in the first image [(x1, y1), (x2, y2), ...].
        p2: List of corresponding points in the second image [(x1', y1'), (x2', y2'), ...].
    """
    # Ensure the images are the same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    canvas1 = np.zeros((height, w1, 3), dtype=np.uint8)
    canvas2 = np.zeros((height, w2, 3), dtype=np.uint8)
    
    canvas1[:h1, :w1, :] = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    canvas2[:h2, :w2, :] = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # Combine images side by side
    combined_image = np.hstack((canvas1, canvas2))
    
    # Offset for the second image
    offset_x = w1
    
    # Plot the points and lines
    for (x1, y1), (x2, y2) in zip(p1, p2):
        # Draw points
        cv2.circle(combined_image, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Green for points in image1
        cv2.circle(combined_image, (int(x2) + offset_x, int(y2)), 5, (0, 0, 255), -1)  # Red for points in image2
        
        # Draw lines linking the points
        cv2.line(combined_image, (int(x1), int(y1)), (int(x2) + offset_x, int(y2)), (0, 255, 0), 1)  # Blue line
    
    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Point Correspondences")
    plt.show()

if __name__ == "__main__":
    xfeat_instance = XFeatWrapper()
    '''
    Da salvare
    path_image1 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/62688623_17b5de833a_o.jpg"
    path_image2 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/62688997_a0cdebb0d1_o.jpg"

    path_image1 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/2429046426_eddd69687b_o.jpg"
    path_image2 :

    data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/215038972_b717b9113b_o.jpg




    '''
    path_image1 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/2429046426_eddd69687b_o.jpg"
    #path_image2 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/2431823248_3776ed43ec_o.jpg"
    #path_image2 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/3968509319_096d953be0_o.jpg"


    image1 = cv2.imread(path_image1)
    

    trasformation= [
        # {
        #     'type': "rotation",
        #     'angle': 45,
        #     'pixel': 0
        # },
        # {
        #     'type': "rotation",
        #     'angle': 90,
        #     'pixel': 0
        # },
        # {
        #     'type': "rotation",
        #     'angle': 180,
        #     'pixel': 0
        # }
    ]
    path = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images"

    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            path_image2 = os.path.join(root, name)
            print(path_image2)
            image2 = cv2.imread(path_image2)
            #p1, p2 = xfeat_instance.inference_xfeat_star_our_version(image1, image2, trasformation, top_k=4092)
            p1, p2 = xfeat_instance.inference_xfeat_star_our_version(imset1=image1, imset2=image2, trasformations=trasformation, top_k=10000)
            p1o, p2o = xfeat_instance.match_xfeat_star_original(image1, image2)
            
            print(len(p1), len(p1o))
            # p1o, p2o = xfeat_instance.inference_xfeat_star_original(image1, image2)
            visualize_comparisons(image1, image2, p1, p2, p1o, p2o)
            print()
            #visualize_correspondences(image1, image2, p1, p2)