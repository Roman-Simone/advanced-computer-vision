import cv2
import argparse
import threading
import numpy as np
from time import time, sleep
from xfeat_wrapper import XFeatWrapper
from accelerated_features.third_party import alike_wrapper as alike


def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
    parser.add_argument('--max_kpts', type=int, default=3_000, help='Maximum number of keypoints.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    parser.add_argument('--dataset-dir', type=str, required=False,
                        default='data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images',
                        help="Path to MegaDepth dataset root")
    parser.add_argument('--matcher', type=str, 
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined", "xfeat-star-clustering" ], 
                        default='xfeat-star',
                        help="Matcher 1 to use")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    parser.add_argument('--method', type=str, 
                        choices=['homography', 'fundamental' ], 
                        default='homography',
                        help="Method for xfeat-refined and xfeat-star-refined (homography or fundamental)")
    
    return parser.parse_args()


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


class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
            self.frame = frame
            sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()

    def get_last_frame(self):
        return self.frame


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.cam)
        self.width = args.width
        self.height = args.height
        self.ref_frame = None
        self.ref_precomp = [[],[]]
        self.corners = [[50, 50], [640-50, 50], [640-50, 480-50], [50, 480-50]]
        self.current_frame = None
        self.H = None
        self.setup_camera()

        #Init frame grabber thread
        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 #avg FPS over this number of frames

        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = XFeatWrapper()
        
        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 3

        self.window_name = "Real-time matching - Press 's' to set the reference frame."

        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        #Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas

    def process(self):
        # Create a blank canvas for the top frame
        top_frame_canvas = self.create_top_frame()

        # Match features and draw matches on the bottom frame
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # Draw warped corners
        if self.H is not None and len(self.corners) > 1:
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        # Stack top and bottom frames vertically on the final canvas
        canvas = np.vstack((top_frame_canvas, bottom_frame))

        cv2.imshow(self.window_name, canvas)

    def match_and_draw(self, ref_frame, current_frame):

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []
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
        # points1, points2 = self.method.match_xfeat_refined(ref_frame, current_frame, method="fundamental")
        points1, points2 = call_matcher(self.args.matcher, self.args, self.method, ref_frame, current_frame, trasformation=trasformation)

        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                self.H = None

            if self.args.method in ["SIFT", "ORB"]:
                good_matches = [m for i,m in enumerate(matches) if inliers[i]]
            else:
                kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        return matched_frame

    def main_loop(self):
        self.current_frame = self.frame_grabber.get_last_frame()
        self.ref_frame = self.current_frame.copy()
        # self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

        while True:
            if self.current_frame is None:
                break

            t0 = time()
            self.process()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()  

            self.current_frame = self.frame_grabber.get_last_frame()

            #Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = MatchingDemo(args = argparser())
    demo.main_loop()
