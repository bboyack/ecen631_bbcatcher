import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def read_ball_centers(file_path):
    centers_L = []
    centers_R = []
    
    with open(file_path, "r") as file:
        lines = file.readlines()

    current_list = None  # Track whether we are reading Left or Right data

    for line in lines:
        line = line.strip()

        if line == "Global Ball Centers Left":
            current_list = centers_L
            continue
        elif line == "Global Ball Centers Right":
            current_list = centers_R
            continue

        if current_list is not None and line:
            # Convert comma-separated values to a tuple of integers
            current_list.append(list(map(int, line.split(","))))

    return centers_L, centers_R


def plot_trajectory(points):
    # Extract X, Y, Z coordinates
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Predict Trajectory fit a polynomial to it
    deg = 2 # Gravity is quadratic so I may as well fit a quadratic polynomial
    poly_Y = np.polynomial.polynomial.Polynomial.fit(Z, Y, deg)
    poly_X = np.polynomial.polynomial.Polynomial.fit(Z, X, deg)

    Z_fit = np.linspace(max(Z), 0, 100)
    Y_fit = poly_Y(Z_fit)
    X_fit = poly_X(Z_fit)

    x0, y0 = poly_X(0), poly_Y(0)
    print(f"Final Position (z=0): ({x0}, {y0})")

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



if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Read ball centers from file
    file_path = "ball_centers/ball_launch1_1.txt"
    centers_L, centers_R = read_ball_centers(file_path)
    centers_L, centers_R = np.array(centers_L)[:, :2], np.array(centers_R)[:, :2]
    centers_L, centers_R = centers_L.astype(np.float32), centers_R.astype(np.float32)
    
    print("Ball centers read from file")
    print("Left:", centers_L)
    print("Right:", centers_R)

    # Estimate 3D trajectory

    points3DL, points3DR, points3D_catcher = trajectory_estimation.run_task(4, cornersL=centers_L, cornersR=centers_R)
    print("3D Trajectory estimated")
    print("Left:\n", points3DL)
    print("Right:\n", points3DR)
    print("Catcher:\n", points3D_catcher)



    # Plot 3D trajectory
    plot_trajectory(points3D_catcher.reshape(-1, 3))