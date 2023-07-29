import math
import cv2
import numpy as np
from time import time
import random
import mediapipe as mp
import matplotlib.pyplot as plt
from tkinter import *
from PIL import *
#from PIL import ImageTK, Image


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

#setting landmarks for lefthand, righthand, leftleg, rightleg
bodyparts = ["left hand","rigt hand","left leg","right leg"]
body_landmarks = {"left hand":19,"rigt hand":20,"left leg":31,"right leg":32}


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

root = Tk()  
def startdisplay():
    limit = 60
    def selected():
        st = clicked.get()
        limit = 60*int(st[0])
        videosetup(limit)
    
    root.title('Rapid Moves')
    root.geometry("800x600")

    #bg = PhotoImage(file = "XFT6.gif")
    '''
    canvas = Canvas(root, width=1000, height=800)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0,0, image=bg, anchor="nw")

    #canvas.create_text(400,250, text='Welcome to Rapid Moves!', font = ("Helvetica", 50), fill="black")

    start = Button(root, text="Start")
    start.config(width = 200, height= 100)
    start = canvas.create_window(400,600, anchor="nw", window=start)
    
    '''
    frame = Frame(root, padx = 100, pady=100, bg='#7B68EE')
    frame.pack(padx=10, pady=10)
    #label = Label(root, image=bg)
    #label.place(x=0,y=0, relwidth = 1, relheight = 1)
    
    text = Label(root, text="Welcome to Rapid Moves!", font = ("Helvetica", 30))
    text.pack(pady = 30)
    rules = Label(root, text="Rules:\n\n1) Touch the coin on the screen with the body part(left hand, right hand, left leg, right leg)\n mentioned on the top of the screen to win a point\n\n2) Touch as many coins as you can within the time limit.\n\n3)Press esc if you want to quit the game.", font = ("Helvetica", 24))
    rules.pack(pady = 30)
    
    time = Label(root, text="Select the time limit", font = ("Helvetica", 24))
    time.pack(pady = 10)
    
    options = ['1 mins', "3 mins", "5 mins"]
    clicked = StringVar()
    clicked.set(options[0])
    drop = OptionMenu(root,clicked, *options)
    drop.config(font=("Helvetica", 25))
    drop.pack(pady=20)

    start = Button(root, text="Start", command = selected)
    start.config(font=("Helvetica", 30))
    start.pack(padx=5)
    
def enddisplay(points):
    root.quit()
    r = Tk()
    r.title('Rapid Moves')
    r.geometry("800x600")
    
    frame = Frame(r, padx = 100, pady=100, bg='#7B68EE')
    frame.pack(padx=10, pady=10)
    label = Label(r, text="Game Over\n\nCongratulations!\n\nYour score is "+str(points), font = ("Helvetica", 30))
    label.pack(pady = 20)
    
    
def videosetup(limit):
    
    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    video = cv2.VideoCapture(0)

    # Create named window for resizing purposes
    
    #cv2.namedWindow('Rapid Moves', cv2.WINDOW_NORMAL)


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
    coinImg = cv2.resize(coinImg, (50,50), interpolation = cv2.INTER_AREA)

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
        frame = cv2.resize(frame, (int(frame_width * (800 / frame_height)), 800))
        frame_height, frame_width, _ =  frame.shape
        #print("frame sizE:", frame_width, frame_height)

        cv2.putText(frame, "Points: "+str(points), (1200,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        
        global s,index,x,y
        #print("touched = ", touched)
        if touched == True:
            s,index,x,y = generate_random_coordinates(frame_height, frame_width)
            print(s,x,y)
            points+=1
            touched = False
        
        ##########################################################################################    
        cv2.putText(frame, "Touch with: "+s, (520, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        
        height, width, channels = coinImg.shape
        offset = np.array((y-25,x-25)) #top-left point from which to insert the smallest image. height first, from the top of the window
        frame[offset[0]:offset[0] + 50, offset[1]:offset[1]+ 50] = coinImg
        
        ##########################################################################################
        #print(s,x,y)    
        # Perform Pose landmark detection.
        frame, landmarks  = detectPose(frame, pose_video, display=False)
        #print(landmarks)
        if landmarks:
            x_bodypart = landmarks[body_landmarks[s]][0]
            y_bodypart = landmarks[body_landmarks[s]][1]
            #print(x_bodypart,y_bodypart)
            
            if x-25 <= x_bodypart and x_bodypart <= x+25 and y-25 <= y_bodypart and y_bodypart <= y+25:
                touched = True

        # Set the time for this frame to the current time.
        time2 = time()

        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:

            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)

            # Write the calculated number of frames per second on the frame.
            #cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.putText(frame, 'Time: '+str(round(time2-time0, 3)), (30, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        if time2-time0 >= limit:
            break
            
        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
        
        cv2.putText(frame, 'Click esc to exit', (910, 750),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
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
    
    cv2.putText(frame, "Game Over\nCongratulations!\n Your total points are: "+str(points), (800,300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
    video.release()
    enddisplay(points)
    # Close the windows.
    cv2.destroyAllWindows()
    


startdisplay()
