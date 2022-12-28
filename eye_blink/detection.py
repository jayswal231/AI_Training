import dlib 
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('/home/mukesh/AI_learning_info/blink_Detection/facial emotion/shape_predictor_68_face_landmarks.dat')

leftStart, leftEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rightStart, rightEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EAR_THRESH = 0.3
counter = 0

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
     #   frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (480, 640))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray, 0)
        
        for (i, rect) in enumerate(faces):
            # print(i, type(rect))

            (x,y,w,h) = face_utils.rect_to_bb(rect)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            frame = cv2.putText(frame, f'Face #{i}', (x, y-10), 1,1, (0,255,0), 2)

            shape = landmark_detector(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x,y) in shape:
                cv2.circle(frame, (x,y), 2, (255,0,0), 1)

            leftEye = shape[42:48]
            rightEye = shape[36:42]

            left_EAR = eye_aspect_ratio(leftEye)
            right_EAR = eye_aspect_ratio(rightEye)

            print(left_EAR, right_EAR)
            if left_EAR < EAR_THRESH or right_EAR < EAR_THRESH:
                counter += 1
                
                if counter >= 10:
                    frame = cv2.putText(frame, "WINK DETECTED!!!!!!!!!", (10, 20), 1, 1, (255,255,0), 4)
            else:
                counter = 0

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()