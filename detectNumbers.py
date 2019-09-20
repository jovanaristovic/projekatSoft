
from functions import *



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


def trebaSabrati(linePlus, x, y, w, h, image_orig, image_bin, tacke_arrayPlava, regions_arrayPlava):

    [(xPlavaD, yPlavaG), (xPlavaG, yPlavaD)] = linePlus

    distancePlava, _, preciCePlavu = pnt2line((x, y), (xPlavaD, yPlavaG), (xPlavaG, yPlavaD))

    if (preciCePlavu == True and distancePlava <= 1):
        tackaPlava = (x, y)
        tacke_arrayPlava.append(tackaPlava)

        regionPlava = image_bin[y:y + h + 1, x:x + w + 1]
        regions_arrayPlava.append([resize_region(regionPlava), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (100, 80, 110), 2)

    return regions_arrayPlava, tacke_arrayPlava


def trebaOduzeti(lineMinus, x, y, w, h, image_orig, image_bin, tacke_arrayZelena, regions_arrayZelena):

    [(xZelenaD, yZelenaG), (xZelenaG, yZelenaD)] = lineMinus
    distanceZelena, _, preciCeZelenu = pnt2line((x, y), (xZelenaD, yZelenaG), (xZelenaG, yZelenaD))

    if (preciCeZelenu == True and distanceZelena <= 1):
        tackaZelena = (x, y)
        tacke_arrayZelena.append(tackaZelena)

        regionZelena = image_bin[y:y + h + 1, x:x + w + 1]
        regions_arrayZelena.append([resize_region(regionZelena), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (100, 80, 110), 2)

    return regions_arrayZelena, tacke_arrayZelena

def select_roi(image_orig,contours_numbers, linePlus, lineMinus):

    img_org = image_orig.copy()
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    image_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_arrayPlava = []
    regions_arrayZelena = []

    tacke_arrayPlava = []
    tacke_arrayZelena = []


    i = 0
    while(i < len(contours_numbers)):

        x, y, w, h = cv2.boundingRect(contours_numbers[i])  # koordinate i velicina granicnog pravougaonika

        img = image_orig.copy()
        cv2.drawContours(img, contours_numbers[i], -1, (0, 0, 255), 1)

        cv2.imshow('img', img)  # samo brojevi
        # print('Distanca od plave linije je: ', distancePlava)


        regions_arrayPlava, tacke_arrayPlava = trebaSabrati(linePlus, x, y, w, h, image_orig, image_bin, tacke_arrayPlava, regions_arrayPlava)
        regions_arrayZelena, tacke_arrayZelena = trebaOduzeti(lineMinus, x, y, w, h, image_orig, image_bin, tacke_arrayZelena, regions_arrayZelena)


        i = i + 1

    # sortirani regioni za plavu liniju
    regions_arrayPlava = sorted(regions_arrayPlava, key=lambda item: item[1][0])
    sorted_regionsPlava = sorted_regionsPlava = [region[0] for region in regions_arrayPlava]

    # sortirani regioni za zelenu liniju
    regions_arrayZelena = sorted(regions_arrayZelena, key=lambda item: item[1][0])
    sorted_regionsZelena = sorted_regionsZelena = [region[0] for region in regions_arrayZelena]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regionsPlava, sorted_regionsZelena, x, y, tacke_arrayPlava, tacke_arrayZelena

