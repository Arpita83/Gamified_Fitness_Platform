import cv2 
import numpy as np 
import time

import pyautogui
import mediapipe as mp

import matplotlib.pyplot as plt 


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


global upperbound 
global lowerbound
global leftbound 
global rightbound

#logic
isInit = False
prevSum = 0
sum_list = np.array([0.0]*5)


#d - move forward
#j - jump
#s - shoot
d_down = False
s_down = False
j=c=False


def findbound(frame_height, frame_width):
        upperbound = frame_height//4
        lowerbound = frame_height*3//5
        leftbound = frame_width//3
        rightbound = frame_width*2//3
        return upperbound, lowerbound, leftbound, rightbound

def findSum(lst):
	sm = 0
	##right wrist       left wrist         nose           left hip         right hip
	sm = lst[16].y*640 + lst[15].y*640 + lst[0].y*640 + lst[23].y*640 + lst[24].y*640
	return sm

def push(sm): 
	global sum_list
	for i in range(3, -1, -1): #reverse loop 3,2,1,0
		sum_list[i+1] = sum_list[i]

	sum_list[0] = abs(sm-prevSum)

def isRunning():
	sm = 0
	for i in sum_list:
		sm = sm + i

	if sm > 100:
		return True 
	return False

def inFrame(lst):
        #print(lst[12][2])
        #right shoulder          left shoulder
        if lst:
                if lst[14].visibility > 0.7 and lst[11].visibility > 0.7:
                        return True
        return False

def isJump(p,upperbound):
	if p < upperbound:
		return True
	return False

def isCrouch(p, lowerbound):
        if p > lowerbound:
                return True
        return False


def isLeft(lst, leftbound):
        #right shoulder
        if lst[12][0] < leftbound:
                return True
        return False

def isRight(lst, rightbound):
        # left shoulder
        if lst[11][0] > rightbound:
                return True
        return False

        
def isShoot(finalres):
	if abs(finalres[15][0] - finalres[16][0]) < 100:
		return True 
	return False

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)
		

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []
    landmarkres = []
    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        landmarkres = results.pose_landmarks.landmark

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        

    # Otherwise
    #else:

        # Return the output image and the found landmarks.
    return output_image, landmarks, landmarkres

global frame_height
global frame_width

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0)

# Create named window for resizing purposes
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)


# Initialize the VideoCapture object to read from a video stored in the disk.
#video = cv2.VideoCapture('media/running.mp4')

# Set video camera size
video.set(3,1280)
#video.set(4,960)
stime = time.time()
isInit = False
#prevSum = 0
#sum_list = np.array([0.0]*5)
    
while video.isOpened():

    # Read a frame.
    ok, frame = video.read()

    # Check if frame is not read properly.
    if not ok:

        # Break the loop.
        break

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame_height, frame_width, _ =  frame.shape
    upperbound, lowerbound, leftbound, rightbound = findbound(frame_height, frame_width)
    frame, landmarks, finalres  = detectPose(frame, pose_video, display=False)

    #main logic
    if finalres and inFrame(finalres):
            # Move to 1300, 800, then click the left mouse button to start the game.
            #if Start():
                #pyautogui.click(x=1300, y=800, button='left')
                
            if not(isInit):
                    prevSum = findSum(finalres)
                    isInit = True

            else:
                    newSum = findSum(finalres)
                    push(newSum)
                        
                    if isJump(landmarks[0][1], upperbound): #press up arrow only if j is not already down
                            cv2.putText(frame, "JUMP DONE", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            #pyautogui.press('up')

                    if isCrouch(landmarks[0][1], lowerbound): #press down arrow only if c is not already down
                            cv2.putText(frame, "CROUCH DONE", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            #pyautogui.press('down')

                    if isLeft(landmarks, leftbound): #press left arrow only if c is not already down
                            cv2.putText(frame, "LEFT DONE", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            #pyautogui.press("left")
                                    

                    if isRight(landmarks, rightbound): #press right arrow only if c is not already down
                            cv2.putText(frame, "RIGHT DONE", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            #pyautogui.press("right")
                                    

                    if isRunning(): ##  running down the d key
                            cv2.putText(frame, "Running", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			    
                    else:
                            cv2.putText(frame, "You are still", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            #click pause

                                
                    prevSum = newSum
    else:# here chek if any key down the make it up
            cv2.putText(frame, "Make Sure full body in frame", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    #drawing lines for the boundaries
    cv2.line(frame, (0,upperbound), (frame_width,upperbound), (255,0,0), 1)
    cv2.line(frame, (0,lowerbound), (frame_width,lowerbound), (255,0,0), 1)
        
    cv2.line(frame, (leftbound,0), (leftbound,frame_height), (255,0,0), 1)
    cv2.line(frame, (rightbound,0), (rightbound,frame_height), (255,0,0), 1)
        
    cv2.imshow('Pose Detection', frame)
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if(k == 27):

        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()



