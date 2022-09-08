import cv2


img = cv2.imread('./train/a1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
thresh=255-thresh;


thinned = cv2.ximgproc.thinning(thresh);
cv2.imshow('thresh', thinned)
