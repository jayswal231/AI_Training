import cv2

video = 0 
cap = cv2.VideoCapture(video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video/testVideo.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    if video != 0 :
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame,(640,480))
        pass
    else :
        frame = cv2.resize(frame,(640,480))
    out.write(frame)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()