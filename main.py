import cv2
import mediapipe as mp
from logger.landmark_logger import LandmarkLogger
from logger.face_landmark_logger import FaceLandmarkLogger
from overlays.dress_overlay import apply_dress
from overlays.tshirt_overlay import apply_tshirt
from overlays.pants_overlay import apply_pants
from overlays.glasses_overlay import apply_glasses
from overlays.shoes_overlay import apply_shoes
from overlays.dress_overlay import apply_dress

from logger.landmark_logger import LandmarkLogger
from logger.face_landmark_logger import FaceLandmarkLogger
from PIL import Image

tshirt = Image.open("assets/tshirt.png").convert("RGBA")
glasses = Image.open("assets/glasses.png").convert("RGBA")
pants = Image.open("assets/pants.png").convert("RGBA")
shoes = Image.open("assets/shoes.png").convert("RGBA")
dress = Image.open("assets/dress.png").convert("RGBA")


# MediaPipe modülleri
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video dosyasını aç (kamera yerine video dosyası)
cap = cv2.VideoCapture('test.mp4')

logger = LandmarkLogger("csv/pose_landmarks.csv")
face_logger = FaceLandmarkLogger("csv/face_landmarks.csv")

frame_count = 0

# MediaPipe Pose yapılandırması
with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video bitti veya kare alınamadı.")
            break

        # RGB'ye çevir
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Poz tahmini
        results = pose.process(image)

        # Görselleştirme için tekrar BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Noktaları çiz
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 1. Landmark'ları çiz
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 2. Landmark'ları CSV'ye kaydet
            logger.log(frame_count, results.pose_landmarks)
            face_logger.log(frame_count, results.pose_landmarks)

            image = apply_tshirt(image, landmarks, tshirt)
            image = apply_pants(image, landmarks, pants)
            image = apply_glasses(image, landmarks, glasses)

        # Görüntüyü göster
        cv2.imshow('Sanal Ayna - Video Üzerinde Pose Takibi', image)

        # 'q' tuşu ile çık
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

face_logger.close()
logger.close()
cap.release()
cv2.destroyAllWindows()
