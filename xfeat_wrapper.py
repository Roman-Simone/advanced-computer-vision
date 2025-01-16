import cv2
import copy
import time
import numpy as np
from scipy.spatial import cKDTree
import accelerated_features.modules.xfeat as xfeat

class XFeatWrapper():
    
    def __init__(self, device = "cuda"):
        self.device = device
        self.xfeat_instance = xfeat.XFeat()

    def detect_feature(self, image):
        '''return:
            Dict: 
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        '''
        output = self.xfeat_instance.detectAndCompute(image, top_k = 4096)
        return output[0]


    def inference_xfeat_star_original(self, image1, image2):
        """
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""
        return self.xfeat_instance.match_xfeat_star(image1, image2)


    def inference_xfeat_original(self, image1, image2):
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
        return self.xfeat_instance.match_xfeat(image1, image2)


    def get_homography(self, type_transformation, image):
        
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
        (h, w) = image.shape[:2]
        rotated_image = cv2.warpPerspective(image, homography_matrix, (w, h))
        return rotated_image

    def check_similarity(self, coord1, coord2, threshold = 5):
        x1, y1 = coord1
        x2, y2 = coord2

        return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold

    def find_similar_points(self, transformed_points, keypoints2, threshold=5):
        """
        Trova gli indici dei punti trasformati simili ai punti in keypoints2.
        
        Args:
            transformed_points (np.ndarray): Punti trasformati, dimensione (n, 2).
            keypoints2 (list): Lista di tuple (x, y) dei keypoint.
            threshold (float): Distanza massima per considerare due punti simili.
        
        Returns:
            list: Indici dei punti in transformed_points simili a quelli in keypoints2.
        """
        # Costruisci un k-d tree per i keypoints di riferimento
        tree = cKDTree(keypoints2)

        # Trova i punti vicini per ogni punto trasformato
        idx_ret = []
        for idx, coord in enumerate(transformed_points[:, :2]):
            # Trova i vicini entro il raggio definito dal threshold
            distances, indices = tree.query(coord, k=1, distance_upper_bound=threshold)
            if distances < threshold:  # Se il vicino più vicino è valido
                idx_ret.append(idx)
        
        return idx_ret

    def unify_features(self, features1, features2, homography):
        
        keypoints1  = copy.deepcopy(features1["keypoints"].cpu().numpy())
        keypoints2 = copy.deepcopy(features2["keypoints"].cpu().numpy())

        # Converti i punti in coordinate omogenee (aggiungi 1 come terzo elemento)
        homogeneous_points = np.hstack([keypoints1, np.ones((keypoints1.shape[0], 1))])

        # Applica la matrice di omografia ai punti
        transformed_points = (homography @ homogeneous_points.T).T

        transformed_points /= transformed_points[:, 2][:, np.newaxis]  # Dividi per il terzo elemento


        # Chiama la funzione
        idx_selected = self.find_similar_points(transformed_points, keypoints2, threshold=5)
      
        keypoints_selected= features1["keypoints"][idx_selected]
        scores_selected= features1["scores"][idx_selected]
        descriptors_selected= features1["descriptors"][idx_selected]

        return {"keypoints": keypoints_selected, 
                "scores": scores_selected, 
                "descriptors": descriptors_selected}


    
    def detect_trasformated_feature(self, image, trasformations):

        features_original = self.detect_feature(image)

        features_filtered = copy.deepcopy(features_original)

        
        for trasformation in trasformations:

            homography = self.get_homography(trasformation, image)

            image_transformed = self.get_image_trasformed(image, homography)

            features_trasformed= self.detect_feature(image_transformed)

            features_filtered = self.unify_features(features_filtered, features_trasformed, homography)
        
        return features_filtered
    
        
    def inference_xfeat_our_version(self, image1, image2, trasformations):
        '''
            Dict: 
                'type': rotation - traslation
                'angle': grades
                'pixel': traslation pixels number
        '''

        # Supponendo che transformed_points e keypoints2 siano già definiti
        # start_time = time.time()

        features_image1 = self.detect_trasformated_feature(image1, trasformations)

        # end_time = time.time()

        # Calcola il tempo trascorso
        # execution_time = end_time - start_time
        # print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
        
        features_image2 = self.detect_feature(image2)

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']


        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, -1)

        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()

        return points1, points2
    
    def match_evaluation(self, img1, img2, top_k = None, nmin_cossim = -1, trasformations= None):

        features_image1 = self.detect_trasformated_feature(img1, trasformations)
        features_image2 = self.detect_trasformated_feature(img2, trasformations)
        # end_time = time.time()

        # Calcola il tempo trascorso
        # execution_time = end_time - start_time
        # print(f"Tempo di esecuzione: {execution_time:.6f} secondi")
        
        # features_image1 = self.detect_feature(image1)
        # features_image2 = self.detect_feature(image2)

        kpts1, descs1 = features_image1['keypoints'], features_image1['descriptors']
        kpts2, descs2 = features_image2['keypoints'], features_image2['descriptors']


        idx0, idx1 = self.xfeat_instance.match(descs1, descs2, min_cossim=nmin_cossim)

        return features_image1['keypoints'][idx0].cpu().numpy(), features_image2['keypoints'][idx1].cpu().numpy()





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