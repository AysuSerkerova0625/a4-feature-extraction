import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

INPUT_PATH = "images/originals/img.jpg"
OUTPUT_DIR = "images/snakes"

clicked_points = []


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_resize_image(path, max_width=700):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        clicked_points.append([event.xdata, event.ydata])
        plt.plot(event.xdata, event.ydata, "ro", markersize=4)
        plt.draw()


def main():
    ensure_dirs()

    image_bgr = load_and_resize_image(INPUT_PATH, max_width=700)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) / 255.0

    smoothed = gaussian(gray, sigma=2.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image_rgb)
    ax.set_title("Click around the object boundary, then close the window")
    fig.canvas.mpl_connect("button_press_event", onclick)

    print("Please click points around the object, then close the window.")
    plt.show()


    if len(clicked_points) < 3:
        print("Not enough points selected.")
        return

    init = np.array(clicked_points)

    snake = active_contour(
        smoothed,
        init,
        alpha=0.02,
        beta=1.0,
        gamma=0.001,
        max_num_iter=800,
        convergence=0.3
    )

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.imshow(image_rgb)
    ax2.plot(init[:, 0], init[:, 1], "--r", lw=2, label="Initial contour")
    ax2.plot(snake[:, 0], snake[:, 1], "-b", lw=2, label="Active contour")
    ax2.legend()
    ax2.set_title("Active Contour Result")

    output_path = os.path.join(OUTPUT_DIR, "active_contour_result.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
