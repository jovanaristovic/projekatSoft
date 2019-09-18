import numpy as np
import os
from keras.models import model_from_json
import cv2
from keras.datasets import mnist
from functions import distance2D, pnt2line, resize_region, image_gray, in_rangeImage, create_ann, train_ann, prepare_for_ann, display_result, convert_output




##############################################    LINIJA      #########################################################

def detectLineHog(img):

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konverzija BGR u RGB
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # slika konvertovana u crno-belu
    ret, t = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)  # primena binarnog tresholda
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # primena Hjuove transformacije sa prosledjenim parametrima
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180 , threshold=100,
                            minLineLength=100, maxLineGap=100)

    xDonje, yDonje, xGornje, yGornje = [], [], [], []  # donja leva, donja desna, gornja leva, gornja desna
    # ima 4 jer je linija deblja
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                xDonje.append(x1)
                yDonje.append(y2)
                xGornje.append(x2)
                yGornje.append(y1)
        xDonje1, yDonje1, xGornje1, yGornje1 = min(xDonje), min(yDonje), max(xGornje), max(yGornje)

        return xDonje1, yGornje1, xGornje1, yDonje1  # vraca koordinate

    # ovu pozivam za pronalazak plave


def findBlueLine2(frame):
    lower = np.array([0, 0, 100], dtype="uint8")
    upper = np.array([50, 50, 255], dtype="uint8")
    blue = in_rangeImage(frame, lower, upper)
    xmin1, ymax1, xmax1, ymin1 = detectLineHog(blue)
    blueLine = (xmin1, ymax1), (xmax1, ymin1)

    # cv2.imshow('blueLine', blue)
    return blueLine

def findGreenLine2(frame):
    lower = np.array([0, 180, 0 ], dtype="uint8")
    upper = np.array([50, 255, 50], dtype="uint8")
    green = in_rangeImage(frame, lower, upper)
    xmin2, ymax2, xmax2, ymin2 = detectLineHog(green)
    greenLine = (xmin2, ymax2), (xmax2, ymin2)

    # cv2.imshow('greenLine', green)
    return greenLine


###########################################  BROJEVI    ###################################################


def detectNumbers(frame):
    img_org = frame.copy()

    gray = image_gray(img_org)
    image_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 25)
    # cv2.imshow('img', image_bin)
    __, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img_org, contours, -1, (0, 0, 255), 1)
    # cv2.imshow('img', img_org)

    contours_numbers = []  # ovde ce biti samo konture koje pripadaju brojevima
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if height < 80 and height > 8 and width > 8:  # uslov da kontura pripada broju
            contours_numbers.append(contour)  # ova kontura pripada broju

    img = frame.copy()
    cv2.drawContours(img, contours_numbers, -1, (0, 0, 255), 1)

    # cv2.imshow('img', img)  # samo brojevi

    return contours_numbers


def select_roi(image_orig,contours_numbers, linePlus, lineMinus):

    img_org = frame.copy()
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    image_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_arrayPlava = []
    regions_arrayZelena = []

    tacke_arrayPlava = []
    tacke_arrayZelena = []

    [(xPlavaD, yPlavaG), (xPlavaG, yPlavaD)] = linePlus

    [(xZelenaD, yZelenaG), (xZelenaG, yZelenaD)] = lineMinus

    # linije
    for contour in contours_numbers:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika

        distancePlava, _, preciCePlavu = pnt2line((x, y), (xPlavaD, yPlavaG), (xPlavaG, yPlavaD))

        img = frame.copy()
        cv2.drawContours(img, contour, -1, (0, 0, 255), 1)

        cv2.imshow('img', img)  # samo brojevi
        # print('Distanca od plave linije je: ', distancePlava)

        distanceZelena, _, preciCeZelenu = pnt2line((x, y), (xZelenaD, yZelenaG), (xZelenaG, yZelenaD))

        area = cv2.contourArea(contour)

        # if ((h >= 10 and w >= 6) and (h <= 28 and w <= 28)):
        # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
        # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

        # sabiranje - plava linija
        if (preciCePlavu == True and distancePlava <= 1):
            tackaPlava = (x, y)
            tacke_arrayPlava.append(tackaPlava)

            regionPlava = image_bin[y:y + h + 1, x:x + w + 1]
            regions_arrayPlava.append([resize_region(regionPlava), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (100, 80, 110), 2)

        # oduzimanje - zelena linija
        if (preciCeZelenu == True and distanceZelena <= 1):
            tackaZelena = (x, y)
            tacke_arrayZelena.append(tackaZelena)

            regionZelena = image_bin[y:y + h + 1, x:x + w + 1]
            regions_arrayZelena.append([resize_region(regionZelena), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (100, 80, 110), 2)

    # sortirani regioni za plavu liniju
    regions_arrayPlava = sorted(regions_arrayPlava, key=lambda item: item[1][0])
    sorted_regionsPlava = sorted_regionsPlava = [region[0] for region in regions_arrayPlava]

    # sortirani regioni za zelenu liniju
    regions_arrayZelena = sorted(regions_arrayZelena, key=lambda item: item[1][0])
    sorted_regionsZelena = sorted_regionsZelena = [region[0] for region in regions_arrayZelena]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regionsPlava, sorted_regionsZelena, x, y, tacke_arrayPlava, tacke_arrayZelena




##########################################################################################################################
def neuronska():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_s = []
    for x in x_train[:10:]:
        ret, slika = cv2.threshold(x, 180, 255, cv2.THRESH_BINARY)
        x_train_s.append(slika)

    ann = create_ann()
    ann = train_ann(ann, np.array(prepare_for_ann(x_train_s), np.float32), convert_output(y_train[:10:]))
    model_json = ann.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    ann.save_weights("model.h5")
    print("Saved model to disk")

    return ann
################################################MAIN###############################

f = open('out.txt', 'a')
f.write('RA 184/2015 Jovana Ristovic\n')
f.write('file sum')
f.close()

if os.path.isfile("model.h5"):

    print("File exists!")
    fileExists=True
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    print("Loaded model from disk")
else:
    print("File does not exist!")
    fileExists=False
    model=neuronska()

for p in range(0, 10):
    # cap = cv2.VideoCapture('videos/video-9.avi')
    NameofVideo = 'video-' + str(p)
    cap = cv2.VideoCapture('videos/' + NameofVideo + '.avi')



    vec_sabrani = []
    vec_oduzeti = []

    vec_uzete_tackeSABIRAJ = []
    vec_uzete_tackeODUZMI = []

    suma = 0
    sumaSabranih = 0
    sumaOduzetih = 0
    while True:

        ret, frame = cap.read()

        if ret == True:

            # detectedLines = {}
            plusLine = findBlueLine2(frame)
            # addLine = detectedLines['plavaLinija']

            minusLine = findGreenLine2(frame)
            # subLine = detectedLines['zelenaLinija']

            contours_numbers = detectNumbers(frame)

            image_orig, sorted_regionsPlava, sorted_regionsZelena, selectRoiX, selectRoiY, tacke_arrayPlava, tacke_arrayZelena = select_roi(frame, contours_numbers, plusLine, minusLine)

            if (len(sorted_regionsPlava) > 0):

                result = model.predict(np.array(prepare_for_ann(sorted_regionsPlava), np.float32))

                rezultat = []
                rezultat = display_result(result)
                # print(rezultat)



                for r in rezultat:
                    for tackaPlava in tacke_arrayPlava:

                        vec_uzete_tackeSABIRAJ.append(tackaPlava)
                        vec_sabrani.append(r)

                        if (len(vec_sabrani) >= 2 and len(vec_uzete_tackeSABIRAJ) >= 2):
                            if (vec_sabrani[len(vec_sabrani) - 2] == vec_sabrani[len(vec_sabrani) - 1]):
                                if (distance2D(vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ) - 2],
                                               vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ) - 1]) <= 3.5):
                                    print('+')
                            else:
                                sumaSabranih += r
                                print('SABIRAM BROJ: ', r)
                        else:
                            print('SABIRAM BROJ: ', r)
                            sumaSabranih += r

            if (len(sorted_regionsZelena) > 0):

                result = model.predict(np.array(prepare_for_ann(sorted_regionsZelena), np.float32))
                rezultat = []
                rezultat = display_result(result)

                for r in rezultat:
                    for tackaZelena in tacke_arrayZelena:

                        vec_uzete_tackeODUZMI.append(tackaZelena)
                        vec_oduzeti.append(r)

                        if (len(vec_oduzeti) >= 2 and len(vec_uzete_tackeODUZMI) >= 2):
                            if (vec_oduzeti[len(vec_oduzeti) - 2] == vec_oduzeti[len(vec_oduzeti) - 1]):
                                if (distance2D(vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI) - 2],
                                               vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI) - 1]) <= 3.5):
                                    print('-')
                            else:
                                sumaOduzetih += r
                                print('ODUZIMAM BROJ: ', r)
                        else:
                            print('ODUZIMAM BROJ: ', r)
                            sumaOduzetih += r

            suma = sumaSabranih - sumaOduzetih
            print(suma)
            # cv2.imshow('dc', frame)



            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    f = open('out.txt', 'a')
    f.write('\n' + NameofVideo + '.avi' + '\t' + str(suma))
    f.close()

