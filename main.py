import cv2
import time
from src.hand_tracker import HandTracker


def main():

    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    cap.set(3,640)
    cap.set(4,480)

    canvas = None
    prev_x, prev_y = 0,0
    smooth_x, smooth_y = 0,0

    brush_thickness = 7
    eraser_thickness = 30
    smoothing = 3

    colors = [
        (0,0,255),    # red
        (255,0,0),    # blue
        (0,255,0),    # green
        (0,255,255),  # yellow
        (255,255,255) # white
    ]

    color_names = ["RED","BLUE","GREEN","YELLOW","WHITE"]
    color_index = 0
    draw_color = colors[color_index]

    locked = False
    save_count = 0

    last_time = 0

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame,1)

        if canvas is None:
            canvas = frame.copy()*0

        timestamp_ms = int(time.time()*1000)

        frame = tracker.find_hands(frame,timestamp_ms)
        landmarks = tracker.find_position(frame,True)

        mode_text = "IDLE"

        if tracker.is_valid_hand(landmarks):

            fingers = tracker.fingers_up(landmarks)

            x,y = landmarks[8][1],landmarks[8][2]

            if smooth_x==0 and smooth_y==0:
                smooth_x, smooth_y = x,y

            smooth_x = int(smooth_x + (x-smooth_x)/smoothing)
            smooth_y = int(smooth_y + (y-smooth_y)/smoothing)

            cv2.circle(frame,(smooth_x,smooth_y),8,(0,0,255),cv2.FILLED)

            # COLOR SELECT (top bar)
            if smooth_y < 60 and fingers[1]==1:

                section = smooth_x//128
                if section < len(colors):
                    color_index = section
                    draw_color = colors[color_index]

            # DRAW
            elif fingers == [0,1,1,0,0] and not locked:

                mode_text="DRAW"

                if prev_x==0 and prev_y==0:
                    prev_x,prev_y=smooth_x,smooth_y

                cv2.line(canvas,(prev_x,prev_y),(smooth_x,smooth_y),draw_color,brush_thickness,cv2.LINE_AA)

                prev_x,prev_y=smooth_x,smooth_y

            # ERASE
            elif fingers == [0,0,1,0,0] and not locked:

                mode_text="ERASE"

                cv2.line(canvas,(prev_x,prev_y),(smooth_x,smooth_y),(0,0,0),eraser_thickness,cv2.LINE_AA)

                prev_x,prev_y=smooth_x,smooth_y

            # CLEAR
            elif fingers == [1,1,1,1,1]:

                mode_text="CLEAR"
                canvas = frame.copy()*0
                prev_x,prev_y = 0,0

            # SAVE
            elif fingers == [1,0,0,0,0]:

                filename=f"drawing_{save_count}.png"
                cv2.imwrite(filename,canvas)
                save_count+=1
                mode_text="SAVED"

            # LOCK / UNLOCK
            elif fingers == [1,0,0,0,1]:

                locked = not locked
                time.sleep(0.5)

            else:
                prev_x,prev_y=0,0

        # merge canvas
        gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame,mask)
        frame = cv2.bitwise_or(frame,canvas)

        # COLOR BAR
        for i,c in enumerate(colors):
            cv2.rectangle(frame,(i*128,0),(i*128+128,60),c,-1)

        cv2.putText(frame,color_names[color_index],(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        if locked:
            cv2.putText(frame,"LOCKED",(500,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # FPS
        current=time.time()
        fps=1/(current-last_time) if last_time!=0 else 0
        last_time=current

        cv2.putText(frame,f"FPS:{int(fps)}",(520,450),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow("AirAuth-SOC",frame)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
