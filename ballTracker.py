import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from estimation3D import Estimate3D

np.set_printoptions(suppress=True)

class BallTracker:
    def __init__(self, first_frame_L, first_frame_R, lower_threshold=20, upper_threshold=200, roi_L=[365,100,30,30], roi_R=[275,100,30,30], debug=False):
        self.background_L = first_frame_L.copy()
        self.background_R = first_frame_R.copy()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.ball_thrown = False
        self.global_ball_center_L_array = []
        self.global_ball_center_R_array = []
        self.local_ball_center_L_array = []
        self.local_ball_center_R_array = []
        self.local_ul_L = [roi_L[0]-roi_L[2], roi_L[1]-roi_L[3]]
        self.local_ul_R = [roi_R[0]-roi_R[2], roi_R[1]-roi_R[3]]
        self.roi_L = roi_L 
        self.roi_R = roi_R
        self.frame_num = 0
        self.prev_circle_L = None
        self.prev_circle_R = None
        self.estimate3d = Estimate3D(debug=debug)

    def get_region_of_interest(self, gray_img, background, roi, upper_left=None):
        gray_img = gray_img.copy()
        background = background.copy()
        cx, cy, dx, dy = roi
        cx, cy, dx, dy = int(cx), int(cy), int(dx), int(dy)
        x = max(cx-dx, 0)
        y = max(cy-dy, 0)
        x2 = min(cx+dx, gray_img.shape[1])
        y2 = min(cy+dy, gray_img.shape[0])

        if upper_left is not None:
            upper_left[0] = x
            upper_left[1] = y
        self.estimate3d.debug_print(f"x: {x}, y: {y}, x2: {x2}, y2: {y2}")
        gray_img = gray_img[y:y2, x:x2]
        background = background[y:y2, x:x2]
        return gray_img, background

    def get_ball_center(self, gray_img, background, lower_bound, upper_bound, show_img=False):
        gray_img = gray_img.copy()
        # Subtract out the background
        diff = cv2.absdiff(background, gray_img)
        # Apply thresholding
        mask = cv2.inRange(diff, lower_bound, upper_bound)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get the largest contour
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the center of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if show_img:
                    cv2.drawContours(gray_img, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(gray_img, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(gray_img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.imshow("Ball Center", gray_img)
                    cv2.waitKey(0)
                return cX, cY, largest_contour
            else:
                return None, None, None
        else:
            return None, None, None
        
    def get_distance_squared(self, x1, y1, x2, y2):
        return ((int(x2)-int(x1))**2 + (int(y2)-int(y1))**2)
    
    def get_ball_center_hough(self, gray_img, min_radius=5, max_radius=20, minDist=50, show_img=False, l_r="L"):
        gray_img = gray_img.copy()
        blurFrame = cv2.GaussianBlur(gray_img, (5, 5), 0) # The bigger the number in the second parameter, the more blurred the image will be. Must be odd
        # dp should be between 1 and 2
        # minDist determines the distance between the centers of the detected circles
        # param1 is sensitivity. If too high, it will not detect circles. If too low, it will detect false circles
        # param2 is the accuracy. Number of edge points that must be in a circle
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=50, param2=8, minRadius=min_radius, maxRadius=max_radius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = circles[0][0]
            # print(circles)
            # print(circles.shape)
            if show_img:
                img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                # draw the circles
                cv2.circle(img, (chosen[0], chosen[1]), chosen[2], (0, 255, 0), 2)
                cv2.circle(img, (chosen[0], chosen[1]), 2, (0, 0, 255), 3)
                # draw dot where previous center was
                # if self.global_ball_center_L_array:
                #     global_prev_center_x, global_prev_center_y, _ = self.global_ball_center_L_array[-1]
                #     local_prev
                cv2.imshow(f"Hough Circle {l_r}", img)
                # if circles.shape[1] > 1:
                #     self.estimate3d.wait_and_close("Hough Circle", close=False)
                # else:
                #     self.estimate3d.wait_and_close("Hough Circle")
                    
            if circles.shape[1] > 1:
                for i in circles[0,1:]:
                    # if self.prev_circle_L is not None:
                        # if self.get_distance_squared(i[0], i[1],self.prev_circle_L[0], self.prev_circle_L[1]) <= self.get_distance_squared(chosen[0], chosen[1], self.prev_circle_L[0], self.prev_circle_L[1]):
                    dist_chosen = self.get_distance_squared(chosen[0], chosen[1], self.roi_L[0], self.roi_L[1])
                    dist_i = self.get_distance_squared(i[0], i[1], self.roi_L[0], self.roi_L[1])
                    self.estimate3d.debug_print(f"dist_chosen: {dist_chosen}, dist_i: {dist_i}")
                    if dist_chosen <= dist_i:
                        chosen = i
                    if show_img:
                        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                        cv2.imshow(f"Hough Circle {l_r}", img)
                        # self.estimate3d.wait_and_close(f"Hough Circle {l_r}", close=False)
                        
                if show_img:
                    cv2.circle(img, (chosen[0], chosen[1]), chosen[2], (255, 0, 0), 2)
                    cv2.imshow(f"Hough Circle {l_r}", img)
                    # self.estimate3d.wait_and_close(f"Hough Circle {l_r}")

            # self.prev_circle_L = chosen
            return (int(chosen[0]), int(chosen[1]), int(chosen[2]))
        else:
            return None
        
    def detect_first_movement(self, gray_img, background, lower_bound, upper_bound, show_img=False):
        ball_moved = False
        gray_img = gray_img.copy()
        # Subtract out the background
        diff = cv2.absdiff(background, gray_img)
        # Apply thresholding
        mask = cv2.inRange(diff, lower_bound, upper_bound)
        if show_img:
            self.estimate3d.display_stereo_img(gray_img, mask, "detect_first_movement mask", wait=False)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If the largest contour is big enough, then the ball has moved
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5:
                ball_moved = True
                if show_img:
                    gray_img = cv2.drawContours(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), [largest_contour], -1, (0, 255, 0), 2)
                    cv2.imshow("First Movement Detection", gray_img)
        
        if show_img:
            self.estimate3d.wait_and_close("detect_first_movement mask")
        return ball_moved

    def save_ball_centers(self, file_name, save_path="ball_centers"):

        # Check to see if folder exists and if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif os.path.exists(f"{save_path}/{file_name}.txt"):
            inp = input(f"\n\"{save_path}/{file_name}.txt\" already exists. Do you want to overwrite it? ([y]/n): ")
            if inp.lower() == "n":
                print("File not saved")
                return
        with open(f"{save_path}/{file_name}.txt", "w") as f:
            f.write("Global Ball Centers Left\n")
            for center in self.global_ball_center_L_array:
                f.write(f"{center[0]}, {center[1]}, {center[2]}, {center[3]}\n")
            f.write("Global Ball Centers Right\n")
            for center in self.global_ball_center_R_array:
                f.write(f"{center[0]}, {center[1]}, {center[2]}, {center[3]}\n")



    def track_ball(self, frameL, frameR, show_img=False):
        self.frame_num += 1
        # print(f"Frame: {self.frame_num}")
        # Wait until the ball is launched
        img_roi_L, background_roi_L = self.get_region_of_interest(frameL, self.background_L, self.roi_L, self.local_ul_L)
        img_roi_R, background_roi_R = self.get_region_of_interest(frameR, self.background_R, self.roi_R, self.local_ul_R)

        if self.ball_thrown == False:
            self.ball_thrown = True # Remove later
            movement = self.detect_first_movement(img_roi_L, background_roi_L, self.lower_threshold, self.upper_threshold, show_img)
            if movement:
                self.ball_thrown = True
                print("Ball Thrown")
                # Update the region of interest
            # else:
            #     self.background_L = frameL
            #     self.background_R = frameR
        else:
            # Threshold the image
            new_imgL = cv2.absdiff(background_roi_L, img_roi_L)
            # self.estimate3d.display_stereo_img(img_roi_L, new_img, "Diff Image", wait=False)
            new_imgL = cv2.inRange(new_imgL, self.lower_threshold, self.upper_threshold)
            if show_img:
                self.estimate3d.display_stereo_img(img_roi_L, new_imgL, "Thresholded Image L", wait=False)
            new_imgR = cv2.absdiff(background_roi_R, img_roi_R)
            new_imgR = cv2.inRange(new_imgR, self.lower_threshold, self.upper_threshold)
            if show_img:
                self.estimate3d.display_stereo_img(img_roi_R, new_imgR, "Thresholded Image R", wait=False)
            # Find the center of the ball
            # minDist = minDist=self.local_ball_center_L_array[-1][2] if len(self.local_ball_center_L_array) > 0 else 50
            centerL = self.get_ball_center_hough(new_imgL, show_img=show_img, minDist=30, l_r="L")
            centerR = self.get_ball_center_hough(new_imgR, show_img=show_img, minDist=30, l_r="R")
            if centerL is not None and centerR is not None:
                # Get the global coordinates
                global_xL = self.local_ul_L[0]+centerL[0]
                global_yL = self.local_ul_L[1]+centerL[1]
                self.global_ball_center_L_array.append((global_xL, global_yL, centerL[2], self.frame_num))
                global_xR = self.local_ul_R[0]+centerR[0]
                global_yR = self.local_ul_R[1]+centerR[1]
                self.global_ball_center_R_array.append((global_xR, global_yR, centerR[2], self.frame_num))
                # Save the local coordinates
                self.local_ball_center_L_array.append(centerL)
                self.roi_L = [global_xL, global_yL, centerL[2]+50, centerL[2]+50]
                self.local_ball_center_R_array.append(centerR)
                self.roi_R = [global_xR, global_yR, centerR[2]+50, centerR[2]+50]
                # print(f"Ball Center: ({global_xL}, {global_yL})")
                # print(f"Local Ball Center: ({centerL[0]}, {centerL[1]})")
                # print(f"ROI: {self.roi_L}")
                # Draw the circle
                # frameL = cv2.cvtColor(frameL, cv2.COLOR_GRAY2BGR)
                # cv2.circle(frameL, (global_xL, global_yL), 2, (100, 200, 0), 3)
                # frameR = cv2.cvtColor(frameR, cv2.COLOR_GRAY2BGR)
                # cv2.circle(frameR, (global_xR, global_yR), 2, (100, 200, 0), 3)
                # stereo_img = self.estimate3d.display_stereo_img(frameL, frameR, "Circle Detected", wait=False)
                # self.estimate3d.wait_and_close("Circle Detected")
                # self.estimate3d.save_img(stereo_img, f"output_images/stereo_imgs_ball_launch1/{self.frame_num}.png")

                
                return (global_xL, global_yL), (global_xR, global_yR)
            else:
                if show_img:
                    cv2.imshow("No Circle Detected", img_roi_L)
                    cv2.imshow("No Circle Detected FrameL", frameL)
                    self.estimate3d.wait_and_close("No Circle Detected")
        return None, None
    

