import cv2 

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

i=0
i+=1
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            x=str(i)
            i+=1
            cv2.imwrite(filename="SSIMAGES/"+x+"saved_img.jpg", img=frame)
            webcam.release()
            img_new = cv2.imread(x+"saved_img.jpg", cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread(x+"saved_img.jpg", cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")

            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=gray)
            print("Image saved!")
            i+=1
            break
        i+=1
        # elif key == ord('q'):
        #     print("Turning off camera.")
        #     webcam.release()
        #     print("Camera off.")
        #     print("Program ended.")
        #     cv2.destroyAllWindows()
        #     break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

