from PIL import Image
import numpy as np

def apply_shoes(image, landmarks, shoe_image):
    h, w, _ = image.shape

    for side in ['left', 'right']:
        if side == 'left':
            ankle = landmarks[27]
            toe = landmarks[31]
        else:
            ankle = landmarks[28]
            toe = landmarks[32]

        # Koordinatları al
        x1 = int(ankle.x * w)
        y1 = int(ankle.y * h)
        x2 = int(toe.x * w)
        y2 = int(toe.y * h)

        # Ayakkabı boyutları
        width = int(abs(x2 - x1) * 3.0) or 1  # genişçe
        height = int(abs(y2 - y1) * 3.0) or 1  # uzun

        resized_shoe = shoe_image.resize((width, height))
        shoe_np = np.array(resized_shoe)

        alpha = shoe_np[:, :, 3] / 255.0
        inv_alpha = 1.0 - alpha

        # Ayakkabıyı bileğin biraz üstünden başlat (gerçekçi görünüm için)
        x_offset = x1 - width // 2
        y_offset = y1 - int(height * 0.3)

        for c in range(3):
            for i in range(height):
                for j in range(width):
                    if 0 <= y_offset + i < h and 0 <= x_offset + j < w:
                        image[y_offset + i, x_offset + j, c] = (
                            alpha[i, j] * shoe_np[i, j, c] +
                            inv_alpha[i, j] * image[y_offset + i, x_offset + j, c]
                        )
    return image
