import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from xfeat_wrapper import XFeatWrapper
from accelerated_features.third_party import alike_wrapper as alike


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


def get_points(matcher_fn, image1=None, image2=None, top_k=4092, trasformations=None, min_cossim=0.9, method='homography'):
    '''
    Get the points from the matcher function
    '''
    if matcher_fn.__name__ == 'match_xfeat_star_original' or matcher_fn.__name__ == 'match_xfeat_original':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k) 
    elif matcher_fn.__name__ == 'match_alike':
        src_pts, dst_pts = matcher_fn(image1, image2) 
    elif matcher_fn.__name__ == 'match_xfeat_trasformed':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, trasformations=trasformations, min_cossim=min_cossim)
    elif matcher_fn.__name__ == matcher_fn.__name__ == 'match_xfeat_star_trasformed':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, trasformations=trasformations)
    elif matcher_fn.__name__ == 'match_xfeat_refined' or matcher_fn.__name__ == 'match_xfeat_star_refined':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, method=method, threshold=90)
    elif matcher_fn.__name__ == 'match_xfeat_star_clustering':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, eps=0.1, min_samples=5)
    else:
        raise ValueError("Invalid matcher")
    
    return src_pts, dst_pts


def call_matcher(modality, args, xfeat_instance, image1, image2, trasformation=None):
    '''
    Call the matcher function based on the modality
    '''

    if modality == 'xfeat':
        print("Running benchmark for XFeat..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_original, image1=image1, image2=image2, top_k=4092)
    elif modality == 'xfeat-star':
        print("Running benchmark for XFeat*..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_original, image1=image1, image2=image2,  top_k=10000)
    elif modality == 'alike':
        print("Running benchmark for alike..")
        return get_points(matcher_fn = alike.match_alike, top_k=None)
    elif modality == 'xfeat-trasformed':
        print("Running benchmark for XFeat with homography trasformation..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_trasformed, image1=image1, image2=image2, top_k=4092, trasformations=trasformation, min_cossim=0.5)   
    elif modality == 'xfeat-star-trasformed':
        print("Running benchmark for XFeat* with homography trasformation..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_trasformed, image1=image1, image2=image2, top_k=10000, trasformations=trasformation, min_cossim=0.5)
    elif modality == 'xfeat-refined':
        print("Running benchmark for XFeat refined..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_refined, image1=image1, image2=image2, top_k=4092, method=args.method)
    elif modality == 'xfeat-star-refined':
        print("Running benchmark for XFeat refined..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_refined, image1=image1, image2=image2, top_k=10000, method=args.method)
    elif modality == 'xfeat-star-clustering':
        print("Running benchmark for XFeat clustering..")
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_clustering, image1=image1, image2=image2, top_k=10000, method=args.method)
    else:
        print("Invalid matcher")


def parse_args():
    parser = argparse.ArgumentParser(description="Run pose benchmark with matcher")
    parser.add_argument('--dataset-dir', type=str, required=False,
                        default='data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images',
                        help="Path to MegaDepth dataset root")
    parser.add_argument('--matcher-1', type=str, 
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined", "xfeat-star-clustering" ], 
                        default='xfeat-star',
                        help="Matcher 1 to use")
    parser.add_argument('--matcher-2', type=str, 
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined", "xfeat-star-clustering" ], 
                        default='xfeat',
                        help="Matcher 1 to use")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    parser.add_argument('--method', type=str, 
                        choices=['homography', 'fundamental' ], 
                        default='homography',
                        help="Method for xfeat-refined and xfeat-star-refined (homography or fundamental)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    PATH = args.dataset_dir
    PATH_IMAGE1 = f"{PATH}/2429046426_eddd69687b_o.jpg"

    xfeat_instance = XFeatWrapper()

    image1 = cv2.imread(PATH_IMAGE1)
    
    trasformation= [
        {
            'type': "rotation",
            'angle': 45,
            'pixel': 0
        },
        {
            'type': "rotation",
            'angle': 90,
            'pixel': 0
        },
        {
            'type': "rotation",
            'angle': 180,
            'pixel': 0
        }
    ]

    for root, dirs, files in os.walk(PATH):
        for name in dirs + files:
            path_image2 = os.path.join(root, name)
            print(path_image2)

            image2 = cv2.imread(path_image2)
            p1, p2 = call_matcher(args.matcher_1, args, xfeat_instance, image1, image2, trasformation=trasformation)
            print("Number of points finds-> ", len(p1))
            p1o, p2o = call_matcher(args.matcher_2, args, xfeat_instance, image1, image2, trasformation=trasformation)
            print("Number of points finds-> ", len(p1o))
            
            visualize_comparisons(image1, image2, p1, p2, p1o, p2o)
