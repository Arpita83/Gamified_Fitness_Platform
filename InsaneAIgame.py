import math
import cv2
import numpy as np
from time import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt



# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

#setting landmarks for lefthand, righthand, leftleg, rightleg
bodyparts = ["left hand","right hand","left leg","right leg"]
body_landmarks = {"left hand":19,"right hand":20,"left leg":31,"right leg":32}


def generate_random_coordinates(h,w):
    bodypart = random.randint(0,3)
    if bodypart > 1:
        x = random.randint(50,w-50)
        y = random.randint(h//2,h-50)
    else:
        x = random.randint(50,w-50)
        y = random.randint(50,h-50)
    #print("rand x=",x,"rand y=",y)
    return (bodyparts[bodypart],bodypart,x,y)


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
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2)
                                  )

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
    return output_image, landmarks
    

def videosetup():

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

    # Initialize a variable to store the time of the previous frame.
    time1 = 0
    touched = True
    points = 0


    coinImg = cv2.imread("COIN.png")
    coinImg = cv2.resize(coinImg, (100,100), interpolation = cv2.INTER_AREA)
    # coinImg = cv2.resize(coinImg,(50,50),fx=0,fy=0, interpolation = cv2.INTER_AREA)
    # import cv2
    # img = cv2.imread("COIN.png")
    # if img is None:
    #     print("Image not loaded. Check the file path.")
    # else:
    #     print("Image loaded successfully. Shape:", img.shape)

    time0 = time()
    # Iterate until the video is accessed successfully.
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
        #print("frame sizE:", frame_width, frame_height)

        cv2.putText(frame, "Points: "+str(points), (950,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
        
        global s,index,x,y
        #print("touched = ", touched)
        if touched == True:
            s,index,x,y = generate_random_coordinates(frame_height, frame_width)
            print(s,x,y)
            points+=1
            touched = False
        
        ##########################################################################################    
        cv2.putText(frame, "Touch with: "+s, (430, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 3)
        
        height, width, channels = coinImg.shape
        offset = np.array((y-50,x-50)) #top-left point from which to insert the smallest image. height first, from the top of the window
        frame[offset[0]:offset[0] + 100, offset[1]:offset[1]+ 100] = coinImg
        
        ##########################################################################################
        #print(s,x,y)    
        # Perform Pose landmark detection.
        frame, landmarks  = detectPose(frame, pose_video, display=False)
        #print(landmarks)
        if landmarks:
            x_bodypart = landmarks[body_landmarks[s]][0]
            y_bodypart = landmarks[body_landmarks[s]][1]
            #print(x_bodypart,y_bodypart)
            
            if x-50 <= x_bodypart and x_bodypart <= x+50 and y-50 <= y_bodypart and y_bodypart <= y+50:
                touched = True

        # Set the time for this frame to the current time.
        time2 = time()

        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:

            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)

            # Write the calculated number of frames per second on the frame.
            #cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.putText(frame, 'Time: '+str(round(time2-time0, 3)), (30, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)

        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
        cv2.putText(frame, 'Click esc to exit', (910, 620),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
        # Display the frame.
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


videosetup()
