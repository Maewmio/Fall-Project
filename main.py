import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') 
video_path = "./Falls/fall-20.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอได้")
    exit()

print("✅ เริ่มการประมวลผลพร้อมระบบ Tracking...")

# สร้าง Dictionary เพื่อเก็บประวัติการร่วง แยกตาม ID ของแต่ละคน!
# ตัวอย่าง: { 1: {'prev_shoulder': 0, 'fall_count': 0}, 2: {'prev_shoulder': 0, 'fall_count': 0} }
people_history = {}

while True:
    success, frame = cap.read()
    if not success:
        break 

    # 1. เปลี่ยนจาก model(frame) เป็น model.track() เพื่อให้ AI แจก ID ให้แต่ละคน
    # persist=True คือให้จำ ID ไว้ตลอดเฟรมถัดๆ ไป
    results = model.track(frame, persist=True, verbose=False) 

    annotated_frame = results[0].plot()

    # 2. นับจำนวนคนในเฟรมปัจจุบัน (นับจากกล่องที่จับได้)
    total_people = len(results[0].boxes)
    # แสดงจำนวนคนบนหน้าจอ (มุมซ้ายบน)
    cv2.putText(annotated_frame, f"People in room: {total_people}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 3. เช็คว่ามีคน และ โมเดลแจก ID ให้คนเหล่านั้นหรือยัง
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        
        # ดึงรายการ ID ของทุกคนที่อยู่ในกล้องตอนนี้
        ids = results[0].boxes.id.int().tolist()
        
        # วนลูปเช็ค "ทุกคน" ที่อยู่ในภาพ (เพื่อหาว่าใครล้มบ้าง)
        for i in range(len(ids)):
            person_id = ids[i]
            keypoints = results[0].keypoints.xy[i] 

            # ถ้าเพิ่งเจอคนนี้ครั้งแรก ให้สร้างประวัติเก็บไว้ให้เขาก่อน
            if person_id not in people_history:
                people_history[person_id] = {'prev_shoulder': 0, 'fall_count': 0}

            if len(keypoints) >= 7: 
                nose_y = float(keypoints[0][1])      
                right_shoulder_y = float(keypoints[6][1]) 
                
                # แสดง ID ไว้บนหัวของคนนั้นๆ
                head_x = int(keypoints[0][0])
                cv2.putText(annotated_frame, f"ID: {person_id}", (head_x - 20, int(nose_y) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # --- เริ่มส่วน FSM Logic (แยกตาม ID) ---
                prev_shoulder_y = people_history[person_id]['prev_shoulder']
                
                if prev_shoulder_y != 0:
                    velocity = right_shoulder_y - prev_shoulder_y 
                    compression = right_shoulder_y - nose_y

                    if velocity > 3.5 and compression < 30: 
                        people_history[person_id]['fall_count'] += 1
                    else:
                        people_history[person_id]['fall_count'] = 0 
                        
                    # ถ้านับการล้มของคนนี้ ครบ 3 เฟรม
                    if people_history[person_id]['fall_count'] >= 3:
                        cv2.putText(annotated_frame, f"PERSON ID {person_id} FELL!!!", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        print(f"🚨 คนที่ ID: {person_id} ล้ม!! (มีคนในห้องทั้งหมด {total_people} คน) 🚨")
                        
                        # ตรงนี้คือจุดที่จะส่ง Firebase!
                        # เช่น โยนข้อมูลขึ้นไปว่า 'person_id': person_id, 'total_people': total_people

                # อัปเดตค่าความสูงไหล่ของคนนี้ เพื่อใช้ในเฟรมถัดไป
                people_history[person_id]['prev_shoulder'] = right_shoulder_y
                # --- จบส่วน FSM Logic ---

    cv2.imshow("YOLOv8 Pose & Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()