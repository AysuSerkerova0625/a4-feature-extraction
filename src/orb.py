import os
import matplotlib.pyplot as plt
import cv2

OUTPUT_DIR = "images/interestPoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# read images
img1 = cv2.imread("images/originals/1.jpg")
img2 = cv2.imread("images/originals/2.jpeg")

if img1 is None or img2 is None:
    raise FileNotFoundError("Check image paths!")

# convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# sort matches
matches = sorted(matches, key=lambda x: x.distance)

# draw matches
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:40],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_path = os.path.join(OUTPUT_DIR, "orb_matches.jpg")
cv2.imwrite(output_path, result)

print(f"Saved result to {output_path}")

# show image
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
