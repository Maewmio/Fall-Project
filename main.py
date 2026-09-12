import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') 
video_path = "./Falls/fall-20.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ไม่สามารถเปิดวิดีโอได้")
    exit()

print("เริ่มกวีดีโอ")

people_history = {}

while True:
    success, frame = cap.read()
    if not success:
        break 
    results = model.track(frame, persist=True, verbose=False) 
    annotated_frame = results[0].plot()
    total_people = len(results[0].boxes)

    cv2.putText(annotated_frame, f"People in room: {total_people}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if results[0].keypoints is not None and results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()

        for i in range(len(ids)):
            person_id = ids[i]
            keypoints = results[0].keypoints.xy[i] 
            if person_id not in people_history:
                people_history[person_id] = {'prev_shoulder': 0, 'fall_count': 0}

            if len(keypoints) >= 7: 
                nose_y = float(keypoints[0][1])      
                right_shoulder_y = float(keypoints[6][1]) 

                head_x = int(keypoints[0][0])
                cv2.putText(annotated_frame, f"ID: {person_id}", (head_x - 20, int(nose_y) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                prev_shoulder_y = people_history[person_id]['prev_shoulder']
                
                if prev_shoulder_y != 0:
                    velocity = right_shoulder_y - prev_shoulder_y 
                    compression = right_shoulder_y - nose_y

                    # ตั้งเงื่อนไข (Threshold) *ตัวเลขพวกนี้ต้องลองปรับจูนเองจากวิดีโอจริง*
                    # สมมติ: ถ้าร่วงเร็วเกิน 5 pixel/เฟรม และ ระยะจมูกกับไหล่หดเหลือน้อยกว่า 30 pixel
                    if velocity > 3.5 and compression < 30: 
                        people_history[person_id]['fall_count'] += 1
                    else:
                        people_history[person_id]['fall_count'] = 0 
                        

                    if people_history[person_id]['fall_count'] >= 3:
                        cv2.putText(annotated_frame, f"PERSON ID {person_id} FELL!!!", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        print(f" คนที่ ID: {person_id} ล้ม!! (มีคนในห้องทั้งหมด {total_people} คน) ")
                        
                people_history[person_id]['prev_shoulder'] = right_shoulder_y


    cv2.imshow("YOLOv8 Pose & Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()