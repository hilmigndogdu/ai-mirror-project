from PIL import Image
import numpy as np

def apply_glasses(image, landmarks, glasses_image):
    h, w, _ = image.shape

    left_eye = landmarks[3]
    right_eye = landmarks[6]

    eye_x1 = int(left_eye.x * w)
    eye_y1 = int(left_eye.y * h)
    eye_x2 = int(right_eye.x * w)
    eye_y2 = int(right_eye.y * h)

    glasses_width = int(abs(eye_x2 - eye_x1) * 2.0)
    glasses_height = int(glasses_width * 0.4)

    center_x = int((eye_x1 + eye_x2) / 2)
    center_y = int((eye_y1 + eye_y2) / 2)

    resized_glasses = glasses_image.resize((glasses_width, glasses_height))
    glasses_np = np.array(resized_glasses)

    alpha_g = glasses_np[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_g

    x_offset = center_x - glasses_width // 2
    y_offset = center_y - glasses_height // 2

    for c in range(3):
        for i in range(glasses_height):
            for j in range(glasses_width):
                if 0 <= y_offset + i < h and 0 <= x_offset + j < w:
                    image[y_offset + i, x_offset + j, c] = (
                        alpha_g[i, j] * glasses_np[i, j, c] +
                        alpha_bg[i, j] * image[y_offset + i, x_offset + j, c]
                    )
    return image
