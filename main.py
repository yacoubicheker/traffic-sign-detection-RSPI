import numpy as np
import cv2
from tensorflow import keras

threshold = 0.75  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffif_sign_model.h5')


def preprocess_img(imgBGR, erode_dilate=True):  # pre-processing fro detect signs in  image.
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin


def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo.item(0) == 0:
        return 'Speed limit (20km/h)'
    elif classNo.item(0) == 1:
        return 'Speed limit (30km/h)'
    elif classNo.item(0) == 2:
        return 'Speed limit (50km/h)'
    elif classNo.item(0) == 3:
        return 'Speed limit (60km/h)'
    elif classNo.item(0) == 4:
        return 'Speed limit (70km/h)'
    elif classNo.item(0) == 5:
        return 'Speed limit (80km/h)'
    elif classNo.item(0) == 6:
        return 'End of speed limit (80km/h)'
    elif classNo.item(0) == 7:
        return 'Speed limit (100km/h)'
    elif classNo.item(0) == 8:
        return 'Speed limit (120km/h)'
    elif classNo.item(0) == 9:
        return 'No passing'
    elif classNo.item(0) == 10:
        return 'No passing for vehicles over 3.5 metric tons'
    elif classNo.item(0) == 11:
        return 'Right-of-way at the next intersection'
    elif classNo.item(0) == 12:
        return 'Priority road'
    elif classNo.item(0) == 13:
        return 'Yield'
    elif classNo.item(0) == 14:
        return 'Stop'
    elif classNo.item(0) == 15:
        return 'No vehicles'
    elif classNo.item(0) == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo.item(0) == 17:
        return 'No entry'
    elif classNo.item(0) == 18:
        return 'General caution'
    elif classNo.item(0) == 19:
        return 'Dangerous curve to the left'
    elif classNo.item(0) == 20:
        return 'Dangerous curve to the right'
    elif classNo.item(0) == 21:
        return 'Double curve'
    elif classNo.item(0) == 22:
        return 'Bumpy road'
    elif classNo.item(0) == 23:
        return 'Slippery road'
    elif classNo.item(0) == 24:
        return 'Road narrows on the right'
    elif classNo.item(0) == 25:
        return 'Road work'
    elif classNo.item(0) == 26:
        return 'Traffic signals'
    elif classNo.item(0) == 27:
        return 'Pedestrians'
    elif classNo.item(0) == 28:
        return 'Children crossing'
    elif classNo.item(0) == 29:
        return 'Bicycles crossing'
    elif classNo.item(0) == 30:
        return 'Beware of ice/snow'
    elif classNo.item(0) == 31:
        return 'Wild animals crossing'
    elif classNo.item(0) == 32:
        return 'End of all speed and passing limits'
    elif classNo.item(0) == 33:
        return 'Turn right ahead'
    elif classNo.item(0) == 34:
        return 'Turn left ahead'
    elif classNo.item(0) == 35:
        return 'Ahead only'
    elif classNo.item(0) == 36:
        return 'Go straight or right'
    elif classNo.item(0) == 37:
        return 'Go straight or left'
    elif classNo.item(0) == 38:
        return 'Keep right'
    elif classNo.item(0) == 39:
        return 'Keep left'
    elif classNo.item(0) == 40:
        return 'Roundabout mandatory'
    elif classNo.item(0) == 41:
        return 'End of no passing'
    elif classNo.item(0) == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while (1):
        ret, img = cap.read()
        img_bin = preprocess_img(img, False)
        cv2.imshow("bin image", img_bin)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)   # get x,y,h and w.
        img_bbx = img.copy()
        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            # rect[2] is width and rect[3] for height
            if rect[2] > 100 and rect[3] > 100:             #only detect those signs whose height and width >100
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            crop_img = np.asarray(img[y1:y2, x1:x2])
            crop_img = cv2.resize(crop_img, (32, 32))
            crop_img = preprocessing(crop_img)
            cv2.imshow("afterprocessing", crop_img)
            crop_img = crop_img.reshape(1, 32, 32, 1)       # (1,32,32) after reshape it become (1,32,32,1)
            predictions = model.predict(crop_img)           # make predicion
            classIndex = np.argmax(model.predict(crop_img),axis=-1)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
                #write class name on the output screen
                cv2.putText(img_bbx, str(classIndex) + " " + str(getClassName(classIndex)), (rect[0], rect[1] - 10),
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                print(getClassName(classIndex))
                # write probability value on the output screen
                cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("detect result", img_bbx)
        if cv2.waitKey(1) & 0xFF == ord('q'):           # q for quit
            break
cap.release()
cv2.destroyAllWindows()