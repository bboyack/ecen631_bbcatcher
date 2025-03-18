import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ballTracker import BallTracker
import time


class BaseballTop(BallTracker):
    def __init__(self, frameL, frameR):
        super().__init__(frameL, frameR)
        # self.estimate3d is inherited from BallTracker
        self.first_pass = True  # Flag to indicate if it's the first pass for initialization
        self.ball_count = 0
        self.no_ball_count = 0
        self.found_ball = True
        self.end = False
        self.varoffset = (0.0, 0.0)  # Placeholder for variable offset
    
    def set_offset(self, offset: tuple):
        self.varoffset = offset  # Ensure it's a 2D array with shape (1, 3)


    def offset(self, pointL, pointR, pointsCatcher):
        # print(f"pointL: {pointL}, pointR: {pointR}, pointsCatcher: {pointsCatcher}")
        offset_array = np.array([[-6.0, -29.0, -25.0]])
        offset_array += np.array([[-13.0, 5.0, 0.0]])  # Placeholder for fixed offset
        # offset_array -= varoffset_array
        # offset_array += np.array([[-19.0, -26.0, 0.0]])
        # print(f"PointL after offset: {pointL + offset_array}, PointR after offset: {pointR + offset_array}, PointsCatcher after offset: {pointsCatcher + offset_array}")
        return pointL + offset_array, pointR + offset_array, pointsCatcher + offset_array


    def run(self, frameL, frameR, show_img=False):
        # Read ball centers
        center_L, center_R = self.track_ball(frameL, frameR)
        if center_L is None or center_R is None:
            # print("No ball detected in one of the frames.")
            return False

        if show_img:
            # Draw centers on the frames
            frameL = cv2.cvtColor(frameL, cv2.COLOR_GRAY2BGR)
            frameR = cv2.cvtColor(frameR, cv2.COLOR_GRAY2BGR)
            # cv2.circle(frameL, tuple(center_L), 2, (0, 255, 0), -1)
            # cv2.circle(frameR, tuple(center_R), 2, (0, 255, 0), -1)
            # cat_frame = np.hstack((frameL, frameR))
            # cv2.imshow("Ball Tracking", cat_frame)
            # self.estimate3d.wait_and_close("Ball Tracking", close=False)

        # Get the 3D position of the ball
        pointsL, pointsR, pointsCatcher = self.estimate3d.get_3D_points(np.array(center_L), np.array(center_R), method="perspective")
        pointsL, pointsR, pointsCatcher = self.offset(pointsL, pointsR, pointsCatcher)
        if self.first_pass:
            # Initialize buffers on the first pass
            self.pointsL_buffer = pointsL
            self.pointsR_buffer = pointsR
            self.pointsCatcher_buffer = pointsCatcher
            self.first_pass = False
        else:
            self.pointsL_buffer = np.vstack((self.pointsL_buffer, pointsL))
            self.pointsR_buffer = np.vstack((self.pointsR_buffer, pointsR))
            self.pointsCatcher_buffer = np.vstack((self.pointsCatcher_buffer, pointsCatcher))

        return True # If successful
    
    def estimate_trajectory(self):
        # Convert to numpy array
        pointsL = np.array(self.pointsL_buffer)
        pointsR = np.array(self.pointsR_buffer)
        pointsCatcher = np.array(self.pointsCatcher_buffer)

        # Fit a line to the points
        # Extract X, Y, Z coordinates
        X, Y, Z = pointsCatcher[:, 0], pointsCatcher[:, 1], pointsCatcher[:, 2]

        # Predict Trajectory fit a polynomial to it
        poly_Y = np.polynomial.polynomial.Polynomial.fit(Z, Y, 2) # Gravity is quadratic so I may as well fit a quadratic polynomial
        poly_X = np.polynomial.polynomial.Polynomial.fit(Z, X, 1)

        x0, y0 = poly_X(0), poly_Y(0)
        x0, y0 = -x0 + self.varoffset[0], -y0 + self.varoffset[1]  # Apply the variable offset
        return x0, y0

    def plot_trajectory(self, points):
        # Extract X, Y, Z coordinates
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

        # Predict Trajectory fit a polynomial to it
        poly_Y = np.polynomial.polynomial.Polynomial.fit(Z, Y, 2) # Gravity is quadratic so I may as well fit a quadratic polynomial
        poly_X = np.polynomial.polynomial.Polynomial.fit(Z, X, 1)

        Z_fit = np.linspace(max(Z), 0, 100)
        Y_fit = poly_Y(Z_fit)
        X_fit = poly_X(Z_fit)

        x0, y0 = poly_X(0), poly_Y(0)
        # print(f"Final Position (z=0): ({x0}, {y0})")

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Side View (Y vs. Z)
        axes[0].scatter(Z, Y, color='red', marker='o', label="Trajectory")
        axes[0].plot(Z_fit, Y_fit, 'b-', label="Trajectory Fit")
        # axes[0].quiver(Z[-2], Y[-2], Z[-1] - Z[-2], Y[-1] - Y[-2], angles='xy', scale_units='xy', scale=1, color='r')
        axes[0].set_xlabel(f"Z (Depth)")
        axes[0].set_ylabel("Y (Height)")
        axes[0].set_title("Side View")
        axes[0].invert_xaxis()  # Make Z decrease from left to right
        axes[0].invert_yaxis()  # Make Y increase upwards
        axes[0].set_aspect('equal')
        axes[0].set_box_aspect(1)
        axes[0].grid()
        axes[0].legend()

        # Top View (X vs. Z)
        axes[1].scatter(Z, X, color='red', marker='o', label="Trajectory")
        axes[1].plot(Z_fit, X_fit, 'b-', label="Trajectory Fit")
        # axes[1].quiver(Z[-2], X[-2], Z[-1] - Z[-2], X[-1] - X[-2], angles='xy', scale_units='xy', scale=1, color='r')
        axes[1].set_xlabel(f"Z (Depth)")
        axes[1].set_ylabel("X (Horizontal)")
        axes[1].set_title("Top View")
        axes[1].invert_xaxis()  # Make Z decrease from left to right
        axes[1].set_aspect('equal')
        axes[1].set_box_aspect(1)
        axes[1].grid()
        axes[1].legend()

        # Adjust layout and show
        plt.tight_layout()
        plt.show()


    def main(self, frameL, frameR):
        if self.end:
            return -1, None, None

        if baseball_top.run(frameL, frameR, show_img=False):
            self.found_ball = True
            self.ball_count += 1
            self.plot_trajectoryno_ball_count = 0
            if (self.ball_count) % 5 == 0 and self.ball_count >= 15 and self.ball_count <=30:
                # Estimate trajectory every 5 frames
                x0, y0 = baseball_top.estimate_trajectory()
                # print(f"Estimated Final Position (z=0): ({x0}, {y0}) - ball_count: {self.ball_count}")

                return self.ball_count, x0, y0
        else:
            # cat_frame = np.hstack((frameL, frameR))
            # cv2.imshow("No ball", cat_frame)
            # wait_and_close("No ball")
            if self.ball_count > 0:
                self.no_ball_count += 1
            # if self.no_ball_count > 5 and ball_count > 20:  # Stop if no ball detected for 5 consecutive frames
            #     # Estimate trajectory every 5 frames
            #     x0, y0 = baseball_top.estimate_trajectory()
            #     # print(f"Estimated Final Position (z=0): ({x0}, {y0}) - ball_count: {self.ball_count}")
            #     baseball_top.plot_trajectory(np.array(baseball_top.pointsCatcher_buffer))
            #     # Reset baseball_top for next sequence
            #     # baseball_top = BaseballTop(frameL, frameR)
            #     # ball_count = 0
                # no_ball_count = 0
                # found_ball = False
                # print("Resetting baseball_top for next sequence...")
        # print(f"Ball Count: {ball_count}, No Ball Count: {no_ball_count}")
        if self.ball_count >= 35 and not self.end:
            self.end = True
            # self.plot_trajectory(np.array(baseball_top.pointsCatcher_buffer))
        return self.ball_count, None, None

def wait_and_close(window_title, time=1, key="q", close=True):
        # Wait until q is pressed or window is closed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break  # Break if the window is closed manually
        if close:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change working directory to script's location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    img_folder_path = "catch_test1"
    if not os.path.exists(img_folder_path):
        raise FileNotFoundError(f"Image folder '{img_folder_path}' does not exist.")
    img_file_path_L = f"{img_folder_path}/L"
    img_file_path_R = f"{img_folder_path}/R"
    img_type = ".png"
    split_text_idx = 0
    # Read in frames
    frames_L = os.listdir(img_file_path_L)
    frames_R = os.listdir(img_file_path_R)
    # Load all images
    frames_L = [f for f in frames_L if f.endswith(img_type)]
    frames_L = sorted(frames_L, key=lambda x: int(os.path.splitext(x)[0][split_text_idx:]))


    frames_R = [f for f in frames_R if f.endswith(img_type)]
    frames_R = sorted(frames_R, key=lambda x: int(os.path.splitext(x)[0][split_text_idx:]))

    print(f"{img_file_path_L}/{frames_L[0]}")
    print(f"{img_file_path_R}/{frames_R[0]}")

    # Get the first frame
    first_frame_L = cv2.imread(f"{img_file_path_L}/{frames_L[0]}", cv2.IMREAD_GRAYSCALE)
    first_frame_R = cv2.imread(f"{img_file_path_R}/{frames_R[0]}", cv2.IMREAD_GRAYSCALE)
    if first_frame_L is None or first_frame_R is None:
        raise FileNotFoundError("Could not read the image files")
    
    baseball_top = BaseballTop(first_frame_L, first_frame_R)

    # Loop through frames
    found_ball = False
    ball_count = 0
    no_ball_count = 0
    for frameL, frameR in zip(frames_L, frames_R):
        frameL = cv2.imread(f"{img_file_path_L}/{frameL}", cv2.IMREAD_GRAYSCALE)
        frameR = cv2.imread(f"{img_file_path_R}/{frameR}", cv2.IMREAD_GRAYSCALE)
        if frameL is None or frameR is None:
            raise FileNotFoundError(f"Could not read the image files: {frameL}, {frameR}")
        
        baseball_top.set_offset((5.6, 7.5))

        ret, x0, y0 = baseball_top.main(frameL, frameR)
        if ret == -1:
            print("End of sequence reached.")
            break
        if x0 is not None and y0 is not None:
            print(f"Estimated Final Position (z=0): ({x0}, {y0}) - ball_count: {ret}")
        