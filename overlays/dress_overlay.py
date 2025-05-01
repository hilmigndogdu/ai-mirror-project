from PIL import Image
import numpy as np

def apply_dress(image, landmarks, dress_image):
    h, w, _ = image.shape

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_knee = landmarks[25]
    right_knee = landmarks[26]

    # Konum ve boyut hesapla
    center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
    top_y = int(min(left_shoulder.y, right_shoulder.y) * h)
    bottom_y = int(max(left_knee.y, right_knee.y) * h)

    dress_width = int(abs(right_shoulder.x - left_shoulder.x) * w * 2.4)
    dress_height = int((bottom_y - top_y) * 1.15)

    resized_dress = dress_image.resize((dress_width, dress_height))
    dress_np = np.array(resized_dress)

    alpha = dress_np[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha

    x_offset = center_x - dress_width // 2
    y_offset = top_y

    for c in range(3):
        for i in range(dress_height):
            for j in range(dress_width):
                if 0 <= y_offset + i < h and 0 <= x_offset + j < w:
                    image[y_offset + i, x_offset + j, c] = (
                        alpha[i, j] * dress_np[i, j, c] +
                        inv_alpha[i, j] * image[y_offset + i, x_offset + j, c]
                    )

    return image
