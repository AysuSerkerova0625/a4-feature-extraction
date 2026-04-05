import os
import cv2
import numpy as np

INPUT_PATH = "images/originals/img.jpg"
EDGE_DIR = "images/edges"
CORNER_DIR = "images/corners"
SHAPE_DIR = "images/shapes"


def ensure_dirs():
    os.makedirs(EDGE_DIR, exist_ok=True)
    os.makedirs(CORNER_DIR, exist_ok=True)
    os.makedirs(SHAPE_DIR, exist_ok=True)


def load_and_resize_image(path, max_width=900):
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


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, blurred, edges


def detect_corners(image, gray):
    result = image.copy()
    gray_f = np.float32(gray)

    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    threshold = 0.01 * harris.max()

    ys, xs = np.where(harris > threshold)
    corners = list(zip(xs, ys))

    for (x, y) in corners:
        cv2.circle(result, (x, y), 5, (0, 0, 255), 2)

    return result, harris, corners


def detect_lines(image, edges):
    result = image.copy()

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result


def detect_circles(image, gray):
    result = image.copy()
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=35,
        minRadius=10,
        maxRadius=150
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c
            cv2.circle(result, (x, y), r, (255, 0, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    return result


def main():
    ensure_dirs()


    image = load_and_resize_image(INPUT_PATH)

    gray, blurred, edges = detect_edges(image)


    corners_img, harris_map, corners = detect_corners(image, gray)
    print(f"Number of detected corners: {len(corners)}")

    lines_img = detect_lines(image, edges)

    circles_img = detect_circles(image, gray)

    cv2.imwrite(os.path.join(EDGE_DIR, "gray.jpg"), gray)
    cv2.imwrite(os.path.join(EDGE_DIR, "blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(EDGE_DIR, "canny_edges.jpg"), edges)

    cv2.imwrite(os.path.join(CORNER_DIR, "harris_corners.jpg"), corners_img)



    cv2.imwrite(os.path.join(SHAPE_DIR, "detected_lines.jpg"), lines_img)
    cv2.imwrite(os.path.join(SHAPE_DIR, "detected_circles.jpg"), circles_img)

    print("Results are saved to images/corners, images/edges, images/shapes")


if __name__ == "__main__":
    main()
