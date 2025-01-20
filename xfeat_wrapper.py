import cv2
import copy
import torch
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from accelerated_features.modules import xfeat

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


    def detect_feature_dense(self, imset, top_k = None, multiscale = True):
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

        imset = self.parse_input(imset)

        output = self.xfeat_instance.detectAndComputeDense(imset, top_k = top_k, multiscale=multiscale)
        
        output_ret = {}
        for key in output.keys():
            if key == "scales":
                output_ret["scores"] = output[key].squeeze(0)
            else:
                output_ret[key] = output[key].squeeze(0)

        return output_ret


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

        imset1 = self.parse_input(imset1)
        imset2 = self.parse_input(imset2)


        return self.xfeat_instance.match_xfeat_star(imset1, imset2, top_k=top_k)


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

        return self.xfeat_instance.match_xfeat(image1, image2, top_k=top_k)
    

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


# UNIFY FEATURES WITH HOMOGRAPHT TRANSFORMATION
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
        trasformed_image = copy.deepcopy(image)
        (h, w) = image.shape[:2]
        trasformed_image = cv2.warpPerspective(trasformed_image, homography_matrix, (w, h))
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

        if len(keypoints.shape) == 3:
            keypoints = keypoints.squeeze(0)

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


    def trasformed_detection_features(self, image, trasformations, merge=False, top_k = None):
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
        features_original = self.detect_feature_sparse(image, top_k=top_k)

        features_filtered = copy.deepcopy(features_original)

        
        for trasformation in trasformations:

            homography = self.get_homography(trasformation, image)

            image_transformed = self.get_image_trasformed(image, homography)

            features_trasformed= self.detect_feature_sparse(image_transformed, top_k=top_k)

            features_filtered = self.unify_features(features_filtered, features_trasformed, homography, merge=merge)
        
        return features_filtered


    def match_xfeat_trasformed(self, image1, image2, trasformations = {}, top_k=4092, min_cossim = None, merge=True):
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
    
        features_image1 = self.trasformed_detection_features(image1, trasformations, merge=merge, top_k=top_k)
        features_image2 = self.trasformed_detection_features(image2, trasformations, merge=merge, top_k=top_k)

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']

        if min_cossim is None: min_cossim = self.min_cossim

        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, min_cossim=min_cossim)

        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        return points1, points2


    def trasformed_detection_features_dense(self, imset, trasformations, merge=True, top_k = None, multiscale = True):
        if top_k is None: top_k = self.top_k

        features_original = self.detect_feature_dense(imset, top_k, multiscale)

        features_filtered = copy.deepcopy(features_original)

        
        for trasformation in trasformations:

            image = copy.deepcopy(imset)

            image = image.permute(0,2,3,1).squeeze(0).cpu().numpy()

            homography = self.get_homography(trasformation, image)

            image_transformed = self.get_image_trasformed(image, homography)

            features_trasformed= self.detect_feature_dense(image_transformed, top_k, multiscale)

            features_filtered = self.unify_features(features_filtered, features_trasformed, homography, merge=merge)
        
        return features_filtered


    def match_xfeat_star_trasformed(self, imset1, imset2, trasformations, top_k = None):

        if top_k == None: top_k = self.top_k
        imset1 = self.parse_input(imset1)
        imset2 = self.parse_input(imset2)

        feature_images1 = self.trasformed_detection_features_dense(imset1, trasformations, merge=True, top_k=top_k, multiscale = True)
        feature_images2 = self.trasformed_detection_features_dense(imset2, trasformations, merge=True, top_k=top_k, multiscale = True)

        feat1 = {}
        feat2 = {}
        for key in feature_images1:
            if key == "scores":
                feat1["scales"] = feature_images1[key].unsqueeze(0)
                feat2["scales"] = feature_images2[key].unsqueeze(0)
            feat1[key] = feature_images1[key].unsqueeze(0)
            feat2[key] = feature_images2[key].unsqueeze(0)

        #Match batches of pairs
        idxs_list = self.xfeat_instance.batch_match(feat1['descriptors'], feat2['descriptors'] )
        B = len(imset1)

        #Refine coarse matches
        #this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(self.xfeat_instance.refine_matches(feat1, feat2, matches = idxs_list, batch_idx=b))

        return matches if B > 1 else (matches[0][:, :2].cpu().detach().numpy(), matches[0][:, 2:].cpu().detach().numpy())

# REFINE FEATURES WITH FUNDAMENTAL AND HOMOGRAPHY
############################################################################################################
    def match_xfeat_refined(self, imset1, imset2, top_k=None, threshold=90, iterations=1, method="homography"):
        '''
            Refine the features with the homography matrix
            input:
                imset1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features
                threshold -> int: distance threshold
                iterations -> int: number of iteration
            return:
                refined_pts1 -> np.ndarray (N, 2): points of the first image
                refined_pts2 -> np.ndarray (N, 2): points of the second image
        '''
        raw_pts1, raw_pts2 = self.match_xfeat_original(imset1, imset2, top_k)

        for i in range(iterations):
            if method == "homography":
                refined_pts1, refined_pts2 = self.filter_by_Homography(raw_pts1, raw_pts2, threshold=threshold)
            elif method == "fundamental":
                refined_pts1, refined_pts2 = self.filter_by_Fundamental(raw_pts1, raw_pts2, threshold=threshold)
            #print(f"Iteration {i+1}: {len(refined_pts1)} matches")
            raw_pts1, raw_pts2 = refined_pts1, refined_pts2  # Update the raw points for the next iteration

        return refined_pts1, refined_pts2


    def match_xfeat_star_refined(self, imset1, imset2, top_k=None, threshold=90, iterations=1, method="homography"):
        '''
            Refine the features with the homography matrix
            input:
                imset1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features
                threshold -> int: distance threshold
                iterations -> int: number of iteration
            return:
                refined_pts1 -> np.ndarray (N, 2): points of the first image
                refined_pts2 -> np.ndarray (N, 2): points of the second image
        '''
        raw_pts1, raw_pts2 = self.match_xfeat_star_original(imset1, imset2, top_k)

        for i in range(iterations):
            if method == "homography":
                refined_pts1, refined_pts2 = self.filter_by_Homography(raw_pts1, raw_pts2, threshold=threshold)
            elif method == "fundamental":
                refined_pts1, refined_pts2 = self.filter_by_Fundamental(raw_pts1, raw_pts2, threshold=threshold)
            #print(f"Iteration {i+1}: {len(refined_pts1)} matches")
            raw_pts1, raw_pts2 = refined_pts1, refined_pts2  # Update the raw points for the next iteration

        return refined_pts1, refined_pts2


    def filter_by_Fundamental(self, pts1, pts2, threshold):
        '''
            Filter the points with the fundamental matrix
            input:
                pts1 -> np.ndarray (N, 2): points of the first image
                pts2 -> np.ndarray (N, 2): points of the second image
                threshold -> int: distance threshold
            return:
                pts1 -> np.ndarray (N, 2): points of the first image
                pts2 -> np.ndarray (N, 2): points of the second image
        '''
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=threshold)
        if mask is None:
            return pts1, pts2
        
        mask = mask.ravel()
        # Filter matches based on inliers
        return pts1[mask == 1], pts2[mask == 1]


    def filter_by_Homography(self, pts1, pts2, threshold):
        '''
            Filter the points with the homography matrix
            input:
                pts1 -> np.ndarray (N, 2): points of the first image
                pts2 -> np.ndarray (N, 2): points of the second image
                threshold -> int: distance threshold
            return:
                pts1 -> np.ndarray (N, 2): points of the first image
                pts2 -> np.ndarray (N, 2): points of the second image
        '''
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
        if mask is None:
            return pts1, pts2
        
        mask = mask.ravel()
        # Filter matches based on inliers
        return pts1[mask == 1], pts2[mask == 1]

# REFINE FEATURES WITH DBSCAN
############################################################################################################
    def filter_with_dbscan(self, features, eps=0.006, min_samples=5):
        '''
            Filter the features with the dbscan algorithm
            input:
                features -> Dict:{keypoints, scores, descriptors}
                eps -> float: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                min_samples -> int: The number of samples in a neighborhood for a point to be considered as a core point.
            return:
                Dict:{keypoints, scores, descriptors}
        ''' 
        # Combine keypoints and descriptors for clustering
        data = features["keypoints"].cpu()
        data_min = data.min(axis=0).values
        data_max = data.max(axis=0).values
        data = (data - data_min) / (data_max - data_min + 1e-8)  # Add a small epsilon to avoid division by zero

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

        # Get valid indices for clustered points
        valid_indices = np.where(clustering.labels_ != -1)[0]
        #print(len(valid_indices))
        if len(valid_indices) == 0:
            print("No valid indices found")
            return features
        # Retain only clustered keypoints and descriptors
        return {"keypoints": features["keypoints"][valid_indices], 
                "scales": features["scores"][valid_indices], 
                "descriptors": features["descriptors"][valid_indices]}


    def match_xfeat_star_clustering(self, imset1, imset2, top_k = None, eps=0.006, min_samples=5):
        '''
            Match the features of two images with the dbscan algorithm
            input:
                imset1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                imset2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features
                eps -> float: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                min_samples -> int: The number of samples in a neighborhood for a point to be considered as a core point.
            return:
                matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
        '''

        if top_k == None: top_k = self.top_k
        imset1 = self.parse_input(imset1)
        imset2 = self.parse_input(imset2)

        feature_images1 = self.detect_feature_dense(imset1, top_k, multiscale = True)
        feature_images2 = self.detect_feature_dense(imset2, top_k, multiscale = True)

        filter_features1 = self.filter_with_dbscan(feature_images1, eps=eps, min_samples=min_samples)
        filter_features2 = self.filter_with_dbscan(feature_images2, eps=eps, min_samples=min_samples)

        feat1 = {}
        feat2 = {}
        for key in filter_features1:
            feat1[key] = filter_features1[key].unsqueeze(0)
            feat2[key] = filter_features2[key].unsqueeze(0)

        #Match batches of pairs
        idxs_list = self.xfeat_instance.batch_match(feat1['descriptors'], feat2['descriptors'] )
        B = len(imset1)

        #Refine coarse matches
        #this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(self.xfeat_instance.refine_matches(feat1, feat2, matches = idxs_list, batch_idx=b))

        return matches if B > 1 else (matches[0][:, :2].cpu().detach().numpy(), matches[0][:, 2:].cpu().detach().numpy())

############################################################################################################
