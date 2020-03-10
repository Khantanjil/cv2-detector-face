# Import library image processing
import cv2

# Read cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread("photo.jpg")

# grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to grayscale image

# Search faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print(type(faces))
print(faces)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# resize image
resized = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
# Open window to show the image
cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
