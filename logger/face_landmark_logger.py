import csv

class FaceLandmarkLogger:
    def __init__(self, filename="face_landmarks.csv"):
        self.file = open(filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z', 'visibility'])

    def log(self, frame_id, landmarks):
        for id in [3, 6]:  # Sadece sol ve sağ göz dışı noktaları
            lm = landmarks.landmark[id]
            self.writer.writerow([frame_id, id, lm.x, lm.y, lm.z, lm.visibility])

    def close(self):
        self.file.close()
