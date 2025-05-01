from PIL import Image
import numpy as np

def apply_tshirt(image, landmarks, tshirt_image):
    h, w, _ = image.shape

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]

    center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
    top_y = int(min(left_shoulder.y, right_shoulder.y) * h)

    shirt_width = int(abs(right_shoulder.x - left_shoulder.x) * w * 2.2)
    shirt_height = int(abs(left_hip.y - left_shoulder.y) * h * 1.2)

    resized_shirt = tshirt_image.resize((shirt_width, shirt_height))
    shirt_np = np.array(resized_shirt)

    alpha_s = shirt_np[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    x_offset = center_x - shirt_width // 2
    y_offset = top_y - 22

    for c in range(3):
        for i in range(shirt_height):
            for j in range(shirt_width):
                if 0 <= y_offset + i < h and 0 <= x_offset + j < w:
                    image[y_offset + i, x_offset + j, c] = (
                        alpha_s[i, j] * shirt_np[i, j, c] +
                        alpha_l[i, j] * image[y_offset + i, x_offset + j, c]
                    )
    return image
