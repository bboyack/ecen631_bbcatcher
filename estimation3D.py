import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import argparse


class Estimate3D:
    def __init__(self, debug=True):
        # Initialize any parameters if needed
        
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (255, 0, 0)
        self.debug = debug

        # Get image shape
        img_shape = (640, 480) # This is the shape of the images used for calibration in (w, h)

        # Get camera parameters
        self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, self.R, self.T, self.E, self.F = self.get_camera_params("../calibration_parameters/left_cam_intr_param.npz", 
                                                                                                                  "../calibration_parameters/right_cam_intr_param.npz", 
                                                                                                                  "../calibration_parameters/stereo_calibration_results.npz")
        print("R: \n", self.R)
        print("T:, \n", self.T)
        # Get undistort and rectify map and parameters
        (self.mapx1, self.mapy1, self.mapx2, self.mapy2), (self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2) = self.get_undistort_and_rectify_map_and_params(img_shape, self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, self.R, self.T)
        # Get undistort and rectify map and parameters but in the right camera frame
        (self.mapx2_r, self.mapy2_r, self.mapx1_r, self.mapy1_r), (self.R2_r, self.R1_r, self.P2_r, self.P1_r, self.Q_r, self.roi2_r, self.roi1_r) = self.get_undistort_and_rectify_map_and_params(img_shape, self.mtx_r, self.dist_r, self.mtx_l, self.dist_l, self.R.T, -self.R.T @ self.T)


    def debug_set(self, val):
        self.debug = val
        print("Debug mode set to ", self.debug)

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def gray2color(self, gray_img):
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    def color2gray(self, color_img):
        return cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    def get_chessboard_corners(self, gray_img, chessboard_size):
            # Termination criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (11, 11)
        zero_zone = (-1, -1)  # no excluded zone
        # Find corners
        ret, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)
        if ret:
            # Refine corner locations
            corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)
            # Get outer 4 corners
            chessboard_4_corners = np.array([
                corners[0],                         # Top Left
                corners[chessboard_size[0] - 1],    # Top Right
                corners[-chessboard_size[0]],       # Bottom Left
                corners[-1]                         # Bottom Right
            ])
            chessboard_4_corners = np.array(chessboard_4_corners, dtype=np.float32).reshape(-1, 1, 2)
            return corners, chessboard_4_corners
        else:
            return None, None
        
    def get_camera_params(self, left_param_file, right_param_file, stereo_param_file):
        # Load calibration results
        left_param = np.load(left_param_file)
        right_param = np.load(right_param_file)
        stereo_param = np.load(stereo_param_file)
        # Get camera parameters
        mtx_l = left_param["mtx"]
        dist_l = left_param["dist"]
        mtx_r = right_param["mtx"]
        dist_r = right_param["dist"]
        R = stereo_param["R"]
        T = stereo_param["T"]
        E = stereo_param["E"]
        F = stereo_param["F"]
        return mtx_l, dist_l, mtx_r, dist_r, R, T, E, F

    def get_undistort_and_rectify_map_and_params(self, img_shape, mtx_l, dist_l, mtx_r, dist_r, R, T):
        # Get rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T)
        # Get rectification map
        mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_32FC1)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_32FC1)
        return (mapx1, mapy1, mapx2, mapy2), (R1, R2, P1, P2, Q, roi1, roi2)

    def undistort_and_rectify_points(self, pointsL, mtxL, distL, RL, PL, pointsR, mtxR, distR, RR, PR):
        pointsL = pointsL.reshape(-1, 1, 2)
        pointsR = pointsR.reshape(-1, 1, 2)
        # Undistort and rectify the image from points
        undistorted_pointsL = cv2.undistortPoints(pointsL, mtxL, distL, R=RL, P=PL)
        undistorted_pointsR = cv2.undistortPoints(pointsR, mtxR, distR, R=RR, P=PR)
        return undistorted_pointsL, undistorted_pointsR

    def undistort_and_rectify_img(self, gray_imgL, gray_imgR, mapxL, mapyL, mapxR, mapyR):
        # Undistort and rectify the image
        undistorted_imgL = cv2.remap(gray_imgL, mapxL, mapyL, cv2.INTER_LINEAR)
        undistorted_imgR = cv2.remap(gray_imgR, mapxR, mapyR, cv2.INTER_LINEAR)
        return undistorted_imgL, undistorted_imgR

    def draw_corners(self, imgL, imgR, cornersL, cornersR, color, with_lines=False, line_color=(255, 0, 0)):
        imgL = imgL.copy()
        imgR = imgR.copy()
        # Check to see if image is gray scale
        if len(imgL.shape) == 2:
            imgL = self.gray2color(imgL)
        if len(imgR.shape) == 2:
            imgR = self.gray2color(imgR)
        # Draw corners
        radius = 5
        thickness = -1
        for corner in cornersL:
            x, y = corner.ravel()
            x, y = int(x), int(y)
            cv2.circle(imgL, (x, y), radius, color, thickness)
            if with_lines:
                cv2.line(imgL, (0, y), (imgL.shape[1], y), line_color, 1)
        for corner in cornersR:
            x, y = corner.ravel()
            x, y = int(x), int(y)
            cv2.circle(imgR, (x, y), radius, color, thickness)
            if with_lines:
                cv2.line(imgR, (0, y), (imgR.shape[1], y), line_color, 1)
        return imgL, imgR

    def display_stereo_img(self, imgL, imgR, title="Stereo Image", wait=True):
        img_concat = cv2.hconcat([imgL, imgR])
        cv2.imshow(title, img_concat)
        if wait:
            self.wait_and_close(title)
        return img_concat

    def wait_and_close(self, window_title, time=1, key="q", close=True):
        # Wait until q is pressed or window is closed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break  # Break if the window is closed manually
        if close:
            cv2.destroyAllWindows()

    def save_img(self, img, save_path):
        # Check to see if path already exists
        save = True
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        elif os.path.exists(save_path):
            inp = input(f"\n\"{save_path}\" already exists. Do you want to overwrite it? ([y]/n): ")
            if inp.lower() == "n":
                save = False
        if save:
            if cv2.imwrite(save_path, img):
                print(f"Image saved to \"{save_path}\"")
            else:
                print(f"Error saving image to \"{save_path}\"")
        else:
            print("Image not saved")

    def do_3D_points_triangulate(self, points2D_L, points2D_R, proj_matrixL, proj_mtxR):
        # Get 3D points
        self.debug_print("points2D_L: ", points2D_L)
        points2D_L = points2D_L.reshape(-1, 2).T
        points2D_R = points2D_R.reshape(-1, 2).T

        self.debug_print("points2D_L reshape: ", points2D_L)
        points3D = cv2.triangulatePoints(proj_matrixL, proj_mtxR, points2D_L, points2D_R)
        points3D = cv2.convertPointsFromHomogeneous(points3D.T).reshape(-1, 3)
        return points3D

    def get_dist_between2points(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def do_perspective_transform(self, points1: np.ndarray, points2: np.ndarray, Q):
        # Calc disparity for points
        self.debug_print("Points1: \n", points1)
        self.debug_print("Points2: \n", points2)
        points1 = points1.reshape(-1, 2)
        points2 = points2.reshape(-1, 2)
        self.debug_print("Points1 reshape: \n", points1)
        self.debug_print("Points2 reshape: \n", points2)
        d = (points1[:,0] - points2[:,0]).reshape(-1, 1)
        self.debug_print("Disparity: \n", d)
        points1d = np.hstack((points1, d))
        points2d = np.hstack((points2, d))
        self.debug_print("Points1d: \n", points1d)
        self.debug_print("Points2d: \n", points2d)
        # Get 3D points
        points13D = cv2.perspectiveTransform(np.astype(points1d.reshape(-1, 1, 3), np.float32), Q)
        points23D = cv2.perspectiveTransform(np.astype(points2d.reshape(-1, 1, 3), np.float32), Q)
        self.debug_print("Points13D: \n", points13D)
        self.debug_print("Points23D: \n", points23D)
        return points13D, points23D

    # Method to check the 3D points
    # Method 1 is to use the R and T matrices to check to see if points 2 will match points 1
    #   pl = R * pr + T
    # Method 2 is to do pl - pr and check to see if the distance is the same as the magnitude of T
    #   pl - pr = T
    #   R should be R_rect for points1 from stereoRectify
    #   T should be from points1 to points2
    # Method 3 is to see if two points on the same image plane will have the same distance as the 3D points given the size of the chessboard square
    #   pl1 - pl2 = 3.88 * 9
    # Points need to be in order of top left, top right, bottom left, bottom right (4x1x3 array)
    def check_points3D(self, points1, points2, method, R=None, T=None, chessboard_shape=(10,7), chessboard_square_size=3.88):
        # Method 1
        if method == 1:
            print("Method 1: pl = R * pr + T")
            self.debug_print("R: \n", R)
            self.debug_print("T: \n", T)
            self.debug_print("Points1: \n", points1)
            self.debug_print("Points2: \n", points2)
            points1_check = (R @ points2.reshape(-1,3).T + T).T.reshape(-1, 1, 3)
            self.debug_print("Points1 Check: \n", points1_check)
            return points1_check
        # Method 2
        elif method == 2:
            print("Method 2: p1 - p2= ||T||")
            T_mag = R @ T
            diff = points1 - points2
            self.debug_print("T_mag: \n", T_mag)
            self.debug_print("Points1 - Points2: \n", diff)
            return diff, T_mag.T
        # Method 3
        elif method == 3:
            dist = self.get_dist_between2points(points1[0], points1[1])
            return dist
        else:
            print("Invalid method")
            return None



    def get_3D_points(self, centerL: np.ndarray, centerR: np.ndarray, method="triangulate"):

        if method=="triangulate":
            # Undistort and rectify the points
            undistorted_pointsL, undistorted_pointsR = self.undistort_and_rectify_points(centerL, self.mtx_l, self.dist_l, self.R1, self.P1, centerR, self.mtx_r, self.dist_r, self.R2, self.P2)

            # Get 3D points Task 1
            points3D_left = self.do_3D_points_triangulate(undistorted_pointsL, undistorted_pointsR, self.P1, self.P2)
            points3D_right = self.do_3D_points_triangulate(undistorted_pointsR, undistorted_pointsL, self.P2_r, self.P1_r)
        else:
            # Get 3D points Task 2
            points3D_left, points3D_right = self.do_perspective_transform(centerL, centerR, self.Q)

            
        points3D_left = points3D_left.reshape(-1, 3)
        points3D_right = points3D_right.reshape(-1, 3)

        # Put left points into the catcher frame, half the distance between the two cameras
        points3D_catcher = points3D_left.copy()
        # print("points3D_catcher before: \n", points3D_catcher)
        points3D_catcher[:, 0] -= 0.5 * (self.R.T @ self.T)[0]
        # print("points3D_catcher after: \n", points3D_catcher)
        return points3D_left, points3D_right, points3D_catcher




if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Estimate the 3D points of a chessboard using stereo calibration")
    parser.add_argument("task_num", type=int, default=1, help="Task number to run (1-3)")
    parser.add_argument("-d", "--debug", action="store_true", help="Disable debug mode (default: enabled)")
    parser.add_argument("-s", "--save_imgs", action="store_true", help="Save images during processing")
    args = parser.parse_args()

    estimate3d = Estimate3D()
    
    task_num = args.task_num
    print(args.debug)
    estimate3d.debug_set(args.debug)
    save_imgs = args.save_imgs

    # Change working directory to script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Print to verify
    print("Current Working Directory:", os.getcwd())
    

    ball_frame_path = "Ball_Launch1"

    estimate3d.get_3D_points(task_num, save_imgs)

    


    
