import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import csv
import pandas as pd
from detectors.yolo_detector import FaceDetector
from recognition.face_embedder import FaceEmbedder

# ✅ Create output folders
os.makedirs("output/faces", exist_ok=True)

# ✅ Load existing visitor log if present
visitor_log_path = "output/visitor_log.csv"
if os.path.exists(visitor_log_path):
    df_existing = pd.read_csv(visitor_log_path)
    known_ids = df_existing["Visitor_ID"].tolist()
    next_id = max(known_ids) + 1 if known_ids else 1
else:
    df_existing = pd.DataFrame(columns=["Visitor_ID", "Timestamp", "Event"])
    known_ids = []
    next_id = 1

# ✅ List to store (embedding, ID) pairs
known_visitors = []
tracked_faces = {}  # Track face last seen time and event status

# ✅ Initialize detector and embedder
detector = FaceDetector(model_path="detectors/yolov8n-face.pt")
embedder = FaceEmbedder()

# ✅ Open CSV in append mode
csv_file = open(visitor_log_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if os.stat(visitor_log_path).st_size == 0:
    csv_writer.writerow(["Visitor_ID", "Timestamp", "Event"])  # Write header if new

# ✅ Helper to log event
def log_event(visitor_id, event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([visitor_id, timestamp, event])
    csv_file.flush()

# ✅ Process webcam and video
video_sources = [0,'sample1.mp4','sample2.mp4','sample3.mp4','sample4.mp4','sample5.mp4','sample6.mp5','sample7.mp4','sample8.mp4','sample9.mp4','sample10.mp4']  # 0 = webcam
out = None

for source in video_sources:
    cap = cv2.VideoCapture(source)
    print(f"🎬 Processing: {'Webcam' if source == 0 else source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if out is None and source != 0:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output/output_video.mp4', fourcc, 20.0, (w, h))

        boxes = detector.detect(frame)
        current_frame_ids = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            embedding = embedder.get_embedding(frame, box)

            if embedding is not None:
                found = False
                for known_embedding, visitor_id in known_visitors:
                    similarity = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
                    if similarity > 0.65:
                        current_frame_ids.append(visitor_id)
                        tracked_faces[visitor_id]["last_seen"] = datetime.now()
                        found = True
                        break

                if not found:
                    visitor_id = next_id
                    next_id += 1
                    known_visitors.append((embedding, visitor_id))
                    known_ids.append(visitor_id)

                    face_img = frame[y1:y2, x1:x2]
                    face_path = f"output/faces/visitor_{visitor_id}.jpg"
                    cv2.imwrite(face_path, face_img)

                    log_event(visitor_id, "entry")
                    tracked_faces[visitor_id] = {"last_seen": datetime.now(), "exited": False}

                    event_type = "entry"
                else:
                    event_type = "reid"

                color = (0, 255, 0) if event_type == "entry" else (0, 0, 255)
                cv2.putText(frame, f"{event_type.upper()} ID {visitor_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ✅ Detect exits
        now = datetime.now()
        for visitor_id in list(tracked_faces.keys()):
            if visitor_id not in current_frame_ids:
                if not tracked_faces[visitor_id]["exited"] and (now - tracked_faces[visitor_id]["last_seen"]).total_seconds() > 3:
                    log_event(visitor_id, "exit")
                    tracked_faces[visitor_id]["exited"] = True

                    cv2.putText(frame, f"EXIT ID {visitor_id}", (10, 60 + 20 * visitor_id),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Show total unique visitors
        cv2.putText(frame, f"Total Visitors: {len(set(known_ids))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if out and source != 0:
            out.write(frame)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()

# ✅ Finalize
if out:
    out.release()
csv_file.close()
cv2.destroyAllWindows()

# ✅ Export to Excel
df = pd.read_csv(visitor_log_path)
df.to_excel("output/visitor_log.xlsx", index=False)

print(f"\n✅ Total Visitors Detected: {len(set(df['Visitor_ID']))}")
print("🎥 Output Video Saved: output/output_video.mp4")
print("📎 Visitor Log Saved: output/visitor_log.csv")
print("📊 Excel Log Saved: output/visitor_log.xlsx")
