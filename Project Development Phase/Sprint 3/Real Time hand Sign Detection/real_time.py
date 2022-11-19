import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
# while True:
#     if(cap.isOpened()):
        
#         break

_, frame = cap.read()
# h,w=500,500
h, w, c = frame.shape
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pixellib
from pixellib.tune_bg import alter_bg

change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
model = load_model('asl_model_97_56.h5')

while True:
    # while(cap.isOpened()):
        _, frame = cap.read()
        # frame = cv2.flip(frame, 1   )
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame
        result = hands.process(frame)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                if(x_max-x_min>y_max-y_min):
                    y_max+=(x_max-x_min-y_max+y_min)//2
                    y_min-=(x_max-x_min-y_max+y_min)//2
                else:
                    x_max+=(y_max-y_min-x_max+x_min)//2
                    x_min-=(y_max-y_min-x_max+x_min)//2
                cv2.rectangle(framergb, (x_min-30, y_min-30), (x_max+30, y_max+30), (0, 255, 0), 2)
                cv2.imwrite("testbox.jpg",framergb[max(0,y_min-30):min(y_max+30,h),max(x_min-30,0):min(x_max+30,w)])
                change_bg.color_bg("testbox.jpg", colors = (255,255,255), output_image_name="colored_bg.jpg")
                
                img = cv2.imread('colored_bg.jpg',2)
                img=cv2.resize(img,(128,128))
                ret, bw_img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV)
                cv2.imwrite("masked.jpeg",bw_img)
                img=image.load_img(r'masked.jpeg',target_size=(128,128),color_mode='grayscale')
                x=image.img_to_array(img)
                x.ndim
                x=np.expand_dims(x,axis=0)
                pred=np.argmax(model.predict(x))
                temp=np.expand_dims(bw_img,axis=0)
                bw_img.shape
                index=['A','B','C','D','E','F','G','H','I']
                print(index[pred])
                # mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
        cv2.imshow("Frame", framergb)

        cv2.waitKey(1)