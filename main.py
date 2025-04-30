import cv2
import mediapipe as mp

# MediaPipe modülleri
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Video dosyasını aç (kamera yerine video dosyası)
cap = cv2.VideoCapture('test.mp4')

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
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Görüntüyü göster
        cv2.imshow('Sanal Ayna - Video Üzerinde Pose Takibi', image)

        # 'q' tuşu ile çık
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
