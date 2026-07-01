import cv2
import mediapipe as mp
import numpy as np
import math
import time

def draw_viewfinder(frame):
    h, w = frame.shape[:2]

    # Size of the middle frame
    box_w = int(w * 0.55)
    box_h = int(h * 0.65)

    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    corner = 35
    thickness = 2
    color = (255, 255, 255)

    # Top Left
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)

    # Top Right
    cv2.line(frame, (x2 - corner, y1), (x2, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)

    # Bottom Left
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(frame, (x1, y2 - corner), (x1, y2), color, thickness)

    # Bottom Right
    cv2.line(frame, (x2 - corner, y2), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y2 - corner), (x2, y2), color, thickness)

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.75, 
        min_tracking_confidence=0.75
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)           

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not open webcam.")
        return

    h, w, c = frame.shape
    mask = np.full((h, w), 255, dtype=np.uint8)

    prev_x, prev_y = None, None
    pen_thickness = 8 
    smoothed_x, smoothed_y = None, None
    smoothing_factor = 0.6  

    last_photo_time = 0
    flash_timer = 0
    is_counting_down = False
    countdown_start = 0

    # gesture system
    l_gesture_frames = 0
    required_frames = 8
    gesture_triggered = False

    print("App started!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-15)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # ---------------- FOG ----------------
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                
                mouth_x = int((top_lip.x + bottom_lip.x) / 2 * w)
                mouth_y = int((top_lip.y + bottom_lip.y) / 2 * h)
                mouth_openness = abs(top_lip.y - bottom_lip.y)
                
                if mouth_openness > 0.06:
                    breath_cloud = np.zeros((h, w), dtype=np.uint8)
                    cv2.ellipse(breath_cloud, (mouth_x, mouth_y), (110, 70), 0, 0, 360, 150, -1)
                    cv2.ellipse(breath_cloud, (mouth_x - 40, mouth_y + 20), (60, 40), -20, 0, 360, 100, -1)
                    cv2.ellipse(breath_cloud, (mouth_x + 40, mouth_y + 20), (60, 40), 20, 0, 360, 50, -1)
                    breath_cloud = cv2.GaussianBlur(breath_cloud, (101, 101), 0)
                    mask = cv2.max(cv2.subtract(mask, breath_cloud), 0)

        # ---------------- HANDS ----------------
        l_gestures_detected = 0
        hands_on_screen = 0

        if hand_results.multi_hand_landmarks:
            hands_on_screen = len(hand_results.multi_hand_landmarks)

            for hand_landmarks in hand_results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                
                index_up = lm[8].y < lm[6].y
                middle_down = lm[12].y > lm[10].y
                ring_down = lm[16].y > lm[14].y
                pinky_down = lm[20].y > lm[18].y
                thumb_out = abs(lm[4].x - lm[9].x) > 0.05

                if index_up and middle_down and ring_down and pinky_down and thumb_out:
                    l_gestures_detected += 1

        # ---------------- GESTURE LOGIC FIXED ----------------
        if l_gestures_detected == 2:
            l_gesture_frames += 1
        else:
            l_gesture_frames = 0
            gesture_triggered = False

        # ONE SHOT TRIGGER (FIXED PROPERLY INSIDE LOOP)
        if (l_gesture_frames >= required_frames 
            and not gesture_triggered 
            and not is_counting_down 
            and (time.time() - last_photo_time) > 2.5
            and hands_on_screen == 2):

            is_counting_down = True
            countdown_start = time.time()
            gesture_triggered = True

        # ---------------- WRITING ----------------
        if hand_results.multi_hand_landmarks and hands_on_screen == 1 and not is_counting_down:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            raw_cx = int(index_finger_tip.x * w)
            raw_cy = int(index_finger_tip.y * h)

            if smoothed_x is None:
                smoothed_x, smoothed_y = raw_cx, raw_cy
            else:
                smoothed_x += smoothing_factor * (raw_cx - smoothed_x)
                smoothed_y += smoothing_factor * (raw_cy - smoothed_y)

            cx, cy = int(smoothed_x), int(smoothed_y)

            if prev_x is not None:
                dist = math.hypot(cx - prev_x, cy - prev_y)
                if dist < 350:
                    cv2.line(mask, (prev_x, prev_y), (cx, cy), 255, pen_thickness)

            prev_x, prev_y = cx, cy
            cv2.circle(mask, (cx, cy), pen_thickness//2, 255, -1)

        else:
            prev_x, prev_y = None, None
            smoothed_x, smoothed_y = None, None

        # ---------------- BLEND ----------------
        fogged_frame = cv2.GaussianBlur(frame, (99, 99), 0)
        white_overlay = np.full_like(frame, (220, 220, 220))
        fogged_frame = cv2.addWeighted(fogged_frame, 0.8, white_overlay, 0.2, 0)

        mask_float = cv2.GaussianBlur(mask, (21, 21), 0).astype(float) / 255.0
        mask_3ch = np.dstack([mask_float] * 3)

        output = (frame * mask_3ch + fogged_frame * (1 - mask_3ch)).astype(np.uint8)
        
        # ---------------- VIEWFINDER ----------------
        # Removed 'and not is_counting_down' so it persists during countdown
        if l_gesture_frames >= required_frames:
            draw_viewfinder(output)

        # ---------------- COUNTDOWN ----------------
        if is_counting_down:
            t = time.time() - countdown_start
            time_left = math.ceil(3 - (t * 1.5))

            if time_left > 0:
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 6
                (tw, th), _ = cv2.getTextSize(str(time_left), font, scale, 20)
                x, y = (w - tw)//2, (h + th)//2

                cv2.putText(output, str(time_left), (x,y), font, scale, (255,255,255), 4)
            else:
                filename = f"Snap_{int(time.time())}.jpg"
                cv2.imwrite(filename, output)
                print("📸 SNAP:", filename)

                last_photo_time = time.time()
                flash_timer = 4
                is_counting_down = False

        if flash_timer > 0:
            output = np.full_like(output, 255)
            flash_timer -= 1

        cv2.imshow("Mirror Camera", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()