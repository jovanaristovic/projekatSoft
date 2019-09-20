import os
from keras.models import model_from_json
from detectLines import *
from neuronska import *
from detectNumbers import *



def sabiranje(model, sorted_regionsPlava, tacke_arrayPlava, sumaSabranih, vec_uzete_tackeSABIRAJ, vec_sabrani):


    result = model.predict(np.array(prepare_for_ann(sorted_regionsPlava), np.float32))

    rezultat = []
    rezultat = display_result(result)
    # print(rezultat)


    i = 0
    while(i<len(rezultat)):

        for tackaPlava in tacke_arrayPlava:

            vec_uzete_tackeSABIRAJ.append(tackaPlava)
            vec_sabrani.append(rezultat[i])

            if (len(vec_sabrani) >= 2 and len(vec_uzete_tackeSABIRAJ) >= 2):
                if (vec_sabrani[len(vec_sabrani) - 2] == vec_sabrani[len(vec_sabrani) - 1]):
                    if (distance2D(vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ) - 2],
                                   vec_uzete_tackeSABIRAJ[len(vec_uzete_tackeSABIRAJ) - 1]) <= 3.5):

                        print('+')
                else:
                    sumaSabranih += rezultat[i]
            else:
                sumaSabranih += rezultat[i]
        i = i + 1

    return sumaSabranih

def oduzimanje(model,sorted_regionsZelena,tacke_arrayZelena, sumaOduzetih, vec_uzete_tackeODUZMI, vec_oduzeti):


    result = model.predict(np.array(prepare_for_ann(sorted_regionsZelena), np.float32))
    rezultat = []
    rezultat = display_result(result)

    i = 0
    while (i < len(rezultat)):

        for tackaZelena in tacke_arrayZelena:

            vec_uzete_tackeODUZMI.append(tackaZelena)
            vec_oduzeti.append(rezultat[i])

            if (len(vec_oduzeti) >= 2 and len(vec_uzete_tackeODUZMI) >= 2):
                if (vec_oduzeti[len(vec_oduzeti) - 2] == vec_oduzeti[len(vec_oduzeti) - 1]):
                    if (distance2D(vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI) - 2],
                                   vec_uzete_tackeODUZMI[len(vec_uzete_tackeODUZMI) - 1]) <= 3.5):

                     print('-')
                else:
                    sumaOduzetih += rezultat[i]
            else:
                sumaOduzetih += rezultat[i]
        i = i + 1

    return sumaOduzetih



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
    NameofVideo = 'video-' + str(p)
    cap = cv2.VideoCapture('videos/' + NameofVideo + '.avi')

    suma = 0
    sumaSabranih = 0
    sumaOduzetih = 0

    vec_uzete_tackeSABIRAJ = []
    vec_uzete_tackeODUZMI = []

    vec_sabrani = []
    vec_oduzeti = []

    while True:

        ret, frame = cap.read()

        if not ret == False:

            plusLine = detectLineHoughBlue(frame)

            minusLine= detectLineHoughGreen(frame)

            contours_numbers = detectNumbers(frame)

            image_orig, sorted_regionsPlava, sorted_regionsZelena, selectRoiX, selectRoiY, tacke_arrayPlava, tacke_arrayZelena = select_roi(frame, contours_numbers, plusLine, minusLine)


            if  (len(sorted_regionsPlava) > 0):
                sumaSabranih = sabiranje(model, sorted_regionsPlava, tacke_arrayPlava, sumaSabranih, vec_uzete_tackeSABIRAJ, vec_sabrani)

            if (len(sorted_regionsZelena) > 0):

                sumaOduzetih = oduzimanje(model,sorted_regionsZelena,tacke_arrayZelena, sumaOduzetih, vec_uzete_tackeODUZMI, vec_oduzeti)

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

