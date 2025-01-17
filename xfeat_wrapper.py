import cv2
import copy
import time
import torch
import numpy as np
from scipy.spatial import cKDTree
import accelerated_features.modules.xfeat as xfeat


class XFeatWrapper():


    def __init__(self, device = "cpu", top_k = 4096, min_cossim = -1):
        self.device = device
        self.xfeat_instance = xfeat.XFeat()
        self.top_k = top_k
        self.min_cossim = min_cossim


    def detect_feature_sparse(self, image, top_k = None):
        '''
        Detects keypoints, descriptors and reliability map for sparse matching (XFeat).
        input: 
            image -> np.ndarray (H,W,C): grayscale or rgb image
        return:
            Dict: 
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        '''
        if top_k is None:
            top_k = self.top_k

        output = self.xfeat_instance.detectAndCompute(image, top_k)
        return output[0]


    def detect_feature_dense(self, imset, top_k = None):
        '''
        Detects keypoints, descriptors and reliability map for semi-dense matching (XFeat Star).
        It works in batched mode because it use different scales of the image.
        input: 
            imset -> torch.Tensor(B, C, H, W): grayscale or rgb image
        return:
            Dict: 
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        '''

        if top_k is None: top_k = self.top_k

        image = self.parse_input(image)

        output = self.xfeat_instance.detectAndComputeDense(imset, top_k = top_k)
        return output[0]


    def match_xfeat_star_original(self, imset1, imset2, top_k = None):
        """
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""

        if top_k is None: top_k = self.top_k

        imset1 = self.parse_input(image1)
        imset2 = self.parse_input(image2)


        return self.xfeat_instance.match_xfeat_star(image1, image2)


    def match_xfeat_original(self, image1, image2, top_k = None):
        """
			Simple extractor and MNN matcher.
			For simplicity it does not support batched mode due to possibly different number of kpts.
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		""" 

        if top_k is None: top_k = self.top_k
        image1 = self.parse_input(image1)
        image2 = self.parse_input(image2)

        return self.xfeat_instance.match_xfeat(image1, image2)


    def parse_input(self, x):
        '''
            Parse the input to the correct format
            return:
                x -> torch.Tensor (B, C, H, W)
        '''
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255

        return x


# FIRST IDEA: UNIFY FEATURES WITH HOMOGRAPHY
############################################################################################################
    def get_homography(self, type_transformation, image):
        '''
            Create the homography matrix for the trasformation given in input
            input:
                type_transformation -> Dict: 
                    'type': rotation - traslation
                    'angle': grades
                    'pixel': traslation pixels number
                image -> np.ndarray (H,W,C): grayscale or rgb image
            return:
                homography_matrix -> np.ndarray (3,3): homography matrix
        '''
        
        homography_matrix = None

        if type_transformation["type"] == "rotation":
            angle = type_transformation["angle"]
            theta = np.radians(angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            (h, w) = image.shape[:2]

            center_x, center_y = w / 2, h / 2
            translation_to_center = np.array([[1, 0, -center_x],
                                            [0, 1, -center_y],
                                            [0, 0, 1]])
            rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]])
            translation_back = np.array([[1, 0, center_x],
                                        [0, 1, center_y],
                                        [0, 0, 1]])
            homography_matrix = translation_back @ rotation_matrix @ translation_to_center
            
        elif type_transformation["type"] == "traslation":
            pass
        else:
            return type_transformation

        return homography_matrix


    def get_image_trasformed(self, image, homography_matrix):
        '''
            Apply the homography matrix to the image
            input:
                image -> np.ndarray (H,W,C): grayscale or rgb image
                homography_matrix -> np.ndarray (3,3): homography matrix
            return:
                trasformed_image -> np.ndarray (H,W,C): grayscale or rgb image
            '''

        (h, w) = image.shape[:2]
        trasformed_image = cv2.warpPerspective(image, homography_matrix, (w, h))
        return trasformed_image


    def filter_points(self, transformed_points, keypoints, threshold=5, merge=False):
        '''
            Filter or unify the points that are near to each other
            input:
                transformed_points -> np.ndarray (N, 2): points trasformed by the homography matrix
                keypoints -> np.ndarray (N, 2): points original to compare
                threshold -> int: distance threshold
                merge -> bool: if True intersect the points, if False unify the points
        
        '''

        tree = cKDTree(keypoints)

        idx_ret = []
        for idx, coord in enumerate(transformed_points[:, :2]):
            distances, indices = tree.query(coord, k=1, distance_upper_bound=threshold)
            if merge:
                if distances < threshold:
                    idx_ret.append(idx)
            else:   
                if distances > threshold:
                    idx_ret.append(idx)
        
        return idx_ret


    def unify_features(self, features1, features2, homography, merge=False):
        '''
            Unify the features of two images (one original and one with homography trasformation)
            input:
                features1 -> Dict:{keypoints, scores, descriptors}
                features2 -> Dict:{keypoints, scores, descriptors}
                homography -> np.ndarray (3,3): homography matrix
                merge -> bool: if True intersect the points, if False unify the points
            return:
                Dict:{keypoints, scores, descriptors}
        '''
        
        keypoints1  = copy.deepcopy(features1["keypoints"].cpu().numpy())
        keypoints2 = copy.deepcopy(features2["keypoints"].cpu().numpy())

        if merge == True:

            homogeneous_points = np.hstack([keypoints1, np.ones((keypoints1.shape[0], 1))])
            transformed_points = (homography @ homogeneous_points.T).T
            transformed_points /= transformed_points[:, 2][:, np.newaxis] 
            idx_selected = self.filter_points(transformed_points, keypoints2, threshold=5, merge=merge)

            keypoints_selected= features1["keypoints"][idx_selected]
            scores_selected= features1["scores"][idx_selected]
            descriptors_selected= features1["descriptors"][idx_selected]

        else:
            homogeneous_points = np.hstack([keypoints2, np.ones((keypoints2.shape[0], 1))])
            transformed_points = (homography @ homogeneous_points.T).T
            transformed_points /= transformed_points[:, 2][:, np.newaxis] 
            idx_selected = self.filter_points(transformed_points, keypoints1, threshold=5, merge=merge)

            keypoints_selected = features1["keypoints"].tolist()  # Convert tensor to list
            scores_selected = features1["scores"].tolist()
            descriptors_selected = features1["descriptors"].tolist()

            for index in idx_selected:
                keypoints_selected.append(features2["keypoints"][index].tolist())
                scores_selected.append(features2["scores"][index].item())  # Convert scalar tensor to Python scalar
                descriptors_selected.append(features2["descriptors"][index].tolist())

            keypoints_selected = torch.tensor(keypoints_selected)
            scores_selected = torch.tensor(scores_selected)
            descriptors_selected = torch.tensor(descriptors_selected)

        return {"keypoints": keypoints_selected, 
                "scores": scores_selected, 
                "descriptors": descriptors_selected}


    def trasformed_detection_features(self, image, trasformations, merge=False):
        '''
            Take an image and apply the trasformations given in input and detect the features unifying or intersecting them
            input:
                image -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]:
                    'type': rotation - traslation
                    'angle': grades
                    'pixel': traslation pixels number
            return:
                Dict:{keypoints, scores, descriptors}

        '''
        features_original = self.detect_feature_sparse(image)

        features_filtered = copy.deepcopy(features_original)

        
        for trasformation in trasformations:

            homography = self.get_homography(trasformation, image)

            image_transformed = self.get_image_trasformed(image, homography)

            features_trasformed= self.detect_feature_sparse(image_transformed)

            features_filtered = self.unify_features(features_filtered, features_trasformed, homography, merge=merge)
        
        return features_filtered


    def inference_xfeat_our_version(self, image1, image2, trasformations, min_cossim = None):
        '''
            Inference of the xfeat algorithm with our version of the trasformation and the match
            input:
                image1 -> np.ndarray (H,W,C): grayscale or rgb image
                image2 -> np.ndarray (H,W,C): grayscale or rgb image
                trasformations -> List[Dict]:
                    'type': rotation - traslation
                    'angle': grades
                    'pixel': traslation pixels number
                min_cossim -> float: minimum cosine similarity to consider a match
            return:
                points1 -> np.ndarray (N, 2): points of the first image
                points2 -> np.ndarray (N, 2): points of the second image
        '''
    
        features_image1 = self.trasformed_detection_features(image1, trasformations, merge=True)
        features_image2 = self.trasformed_detection_features(image2, trasformations, merge=True)

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']

        if min_cossim is None: min_cossim = self.min_cossim

        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, min_cossim=min_cossim)

        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        return points1, points2
############################################################################################################


'''
Image Input -> Trasformation (Rotation, Translataion, known homography) -> Trasfromed Image
Image Input -> XFeat -> Feature Detection 
Trasformed Image -> XFeat -> Feature Detection 

Step choose common feature

on retained_feature -> match_feature


'''


if __name__ == "__main__":
    path_image1 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/29307281_d7872975e2_o.jpg"
    path_image2 = "data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/50646217_c352086389_o.jpg"
    image1 = cv2.imread(path_image1)
    image2 = cv2.imread(path_image2)
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
    

    xfeat_instance = XFeatWrapper()

    xfeat_instance.inference_xfeat_our_version(image1, image2, trasformation)



    # output = xfeat_instance.detect_feature(image1)
    # pts1, pts2 = xfeat_instance.inference_xfeat_original(image1, image2)
    # pts1_our, pts2_our = xfeat_instance.inference_xfeat_our_version(image1, image2)
    # pts1_star, pts2_star = xfeat_instance.inference_xfeat_star_original(image1, image2)
    # print(pts1)
    # print()