import csv

class LandmarkLogger:
    def __init__(self, filename="pose_landmarks.csv"):
        self.file = open(filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z', 'visibility'])

    def log(self, frame_id, landmarks):
        for id, lm in enumerate(landmarks.landmark):
            self.writer.writerow([frame_id, id, lm.x, lm.y, lm.z, lm.visibility])

    def close(self):
        self.file.close()
