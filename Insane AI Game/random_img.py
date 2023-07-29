import cv2
img = cv2.imread("/Insane AI Game/COIN.png")
if img is None:
    print("Image not loaded. Check the file path.")
else:
    print("Image loaded successfully. Shape:", img.shape)