
from functions import *

def detectLineHoughBlue(frame):

    blue = in_rangeImage(frame, np.array([0, 0, 100], dtype="uint8"), np.array([50, 50, 255], dtype="uint8"))

    rgb_image = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)  # konverzija BGR u RGB
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # slika konvertovana u crno-belu
    ret, t = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)  # primena binarnog tresholda
    edges = cv2.Canny(blue, 50, 150, apertureSize=3)

    # primena Hjuove transformacije sa prosledjenim parametrima
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180 , threshold=100,
                            minLineLength=100, maxLineGap=100)

    xDonje, yDonje, xGornje, yGornje = [], [], [], []  # donja leva, donja desna, gornja leva, gornja desna
    # ima 4 jer je linija deblja
    if lines is not None:
        i = 0
        while(i < len(lines) ):
        # for line in lines:

            for x1, y1, x2, y2 in lines[i]:
                xDonje.append(x1)
                yDonje.append(y2)
                xGornje.append(x2)
                yGornje.append(y1)
            i = i + 1

        xDonje1, yDonje1, xGornje1, yGornje1 = min(xDonje), min(yDonje), max(xGornje), max(yGornje)

        return (xDonje1, yGornje1), (xGornje1, yDonje1)  # vraca koordinate

    # ovu pozivam za pronalazak plave


def detectLineHoughGreen(frame):

    green = in_rangeImage(frame,  np.array([0, 180, 0 ], dtype="uint8"), np.array([50, 255, 50], dtype="uint8"))


    rgb_image = cv2.cvtColor(green, cv2.COLOR_BGR2RGB)  # konverzija BGR u RGB
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # slika konvertovana u crno-belu
    ret, t = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)  # primena binarnog tresholda
    edges = cv2.Canny(green, 50, 150, apertureSize=3)

    # primena Hjuove transformacije sa prosledjenim parametrima
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180 , threshold=100,
                            minLineLength=100, maxLineGap=100)

    xDonje, yDonje, xGornje, yGornje = [], [], [], []  # donja leva, donja desna, gornja leva, gornja desna
    # ima 4 jer je linija deblja
    if lines is not None:
        i = 0
        while(i < len(lines) ):
        # for line in lines:

            for x1, y1, x2, y2 in lines[i]:
                xDonje.append(x1)
                yDonje.append(y2)
                xGornje.append(x2)
                yGornje.append(y1)
            i = i + 1

        xDonje1, yDonje1, xGornje1, yGornje1 = min(xDonje), min(yDonje), max(xGornje), max(yGornje)

        return (xDonje1, yGornje1), (xGornje1, yDonje1)  # vraca koordinate

