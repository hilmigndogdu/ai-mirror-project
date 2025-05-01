from PIL import Image
import numpy as np

def apply_pants(image, landmarks, pants_image):
    h, w, _ = image.shape

    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]

    # Merkez belirleme
    center_x = int((left_hip.x + right_hip.x) / 2 * w)
    top_y = int(min(left_hip.y, right_hip.y) * h)
    bottom_y = int(max(left_ankle.y, right_ankle.y) * h)

    pants_width = int(abs(right_hip.x - left_hip.x) * w * 3)
    pants_height = int((bottom_y - top_y) * 1.2)  # Oranı artırdık




    resized_pants = pants_image.resize((pants_width, pants_height))
    pants_np = np.array(resized_pants)

    alpha = pants_np[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha

    x_offset = center_x - pants_width // 2
    y_offset = top_y - 20

    for c in range(3):
        for i in range(pants_height):
            for j in range(pants_width):
                if 0 <= y_offset + i < h and 0 <= x_offset + j < w:
                    image[y_offset + i, x_offset + j, c] = (
                        alpha[i, j] * pants_np[i, j, c] +
                        inv_alpha[i, j] * image[y_offset + i, x_offset + j, c]
                    )
    return image
