import threading

import numpy as np
import matplotlib.path as mplPath
import math
import pyautogui
import PIL
import cv2


def SortByLine(array,returnIndexes=False):
    """[[0, 1],
        [3, 20],
        [6, 3],
        [2, 4],
        [3, 5],
        [1, 6]]

       [[0, 1],
        [1, 6],
        [2, 4],
        [3, 5],
        [3, 20],
        [6, 3]]"""
    #MajorMoxIndex = UsefulMethods.SortByLine(box,returnIndexes=True)[-1]
    returned = np.lexsort(np.fliplr(array).transpose())
    if returnIndexes:
        return returned
    else:
        return array[returned]
def CombineAllElements(arrayA,arrayB):
    #np.array(np.stack(np.meshgrid([1, 0], [1, 0], [1, 0]), -1).reshape(-1, 3))
    #arrayA = (0,1)
    #arrayB = (0,1)
    #[[0 0]
    #[1 0]
    #[0 1]
    #[1 1]]
    return np.array(np.stack(np.meshgrid(arrayA, arrayB), -1).reshape(-1, 2))
def IsInside(polygon,point):
    return mplPath.Path(np.array(polygon)).contains_point(point)
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
def ConvertTextToList(text,divisor):
    lista = []
    b=0
    for i in range(text.count(divisor)):
        c = text.index(divisor,b)
        lista.append(text[b:c])
        b=c+len(divisor)
    return lista
def ChangeArrayIndexes(array,newOrder):
    #array: ('A','B','C')
    #newOrder: (2,0,1)
    #return: ("C","A","B")
    orderer = []
    for i in newOrder:
        orderer.append(array[i])
    return orderer
def GetIndexOfLine(matrix,line):
    #[[1,2]
    #[1,2]
    #[1,1]
    #[3,6]
    #[6,3]
    #[7,8]
    #[9,9]]
    #line = [0 1]
    aux = np.where(matrix==line,matrix.copy(),0)
    aux = np.sum(aux,axis=1)
    return np.array(np.where(aux==np.sum(line)))[0]

def GetCentersOfThreshold(hsvimage,hsvRange,minAreaFigure=10,maxAreaFigure=math.inf,sortArea=False):
    imagem_threshold = cv2.inRange(hsvimage, hsvRange[0], hsvRange[1])  # GreenHSVRange[0], GreenHSVRange[1])
    contours, hierarchy = cv2.findContours(imagem_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(imagem, contours, 0, (0, 0, 0), 3)
    Centers = [[],[]] #centros = (centrosX,centrosY)
    #print("     >> ",len(contours))
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        areas.append(area)
        #print("useGreater: ",sortArea)
        if (area>minAreaFigure and area<maxAreaFigure) or sortArea==True:
            M = cv2.moments(c)
            if M["m00"] != 0:
                    Centers[0].append(int(M["m10"] / M["m00"]))
                    Centers[1].append(int(M["m01"] / M["m00"]))
            else:
                Centers[0].append(0)
                Centers[1].append(0)
    if sortArea:
        Centers = np.array(Centers).transpose()
        #print("Centers:\n",Centers,"\n\n\n\n\n")
        teste = np.insert(Centers,0,np.array(areas),axis=1)
        teste = SortByLine(teste)
        return teste
    else:
        return Centers
def PyAutoGui_To_Cv2(imagem):
    imagem.save("auxiliar.png")
    del imagem
    imagem = cv2.imread("auxiliar.png")
    return imagem
def PrintScreen(rectT):
    (x1,y1) = rectT[0]
    (x2,y2) = rectT[1]
    imagem = pyautogui.screenshot(region=(x1,y1,x2-x1,y2-y1))
    imagem.save("auxiliar.png")
    del imagem
    imagem = PIL.Image.open("auxiliar.png")
    return imagem


def GetDistancesInMap(xyArray_vezes_map, barrierElement):
    aux = np.array(np.where(xyArray_vezes_map == barrierElement))
    return aux[1, np.unique(aux[0, :], return_index=True)[1]]
def Invert01Matrix(array):
    arrayaux = np.where(array!=1,array,2)
    arrayaux = np.where(arrayaux!=0,arrayaux,1)
    return np.where(arrayaux!=2,arrayaux,0)
def MakeBordersIn2DArray(array,borderElement):
    copyarray = np.copy(array)
    copyarray = np.insert(copyarray,array.shape[0],borderElement,axis=0)
    copyarray = np.insert(copyarray, array.shape[1], borderElement, axis=1)
    copyarray = np.insert(copyarray, 0, borderElement, axis=0)
    copyarray = np.insert(copyarray, 0, borderElement, axis=1)
    return copyarray
def GetPositionInArray(element,array):
    arraynp = array == element
    x = np.where(np.sum(np.array(arraynp), axis=1) == 1)[0][0]
    y = np.where(np.sum(np.array(arraynp), axis=0) == 1)[0][0]
    return (x,y)
def GetDirectionsXY(array,xarray,yarray,quantIndividuos):
    Ydirections = (np.resize(array,(quantIndividuos, array.shape[0], array.shape[1]))[:,yarray[:],:])[0, :, :]
    Xdirections = np.matrix.transpose((np.resize(array, (quantIndividuos, array.shape[0], array.shape[1]))[:, :, xarray[:]])[0, :, :])
    return Xdirections, Ydirections
def GetModel01(qInds,shapes,indexes):
    array = np.flip(np.resize(np.arange(1,shapes+1),(qInds,shapes)),axis=0)
    #print(array,"\n------")
    indexes = np.transpose(np.resize(indexes,(shapes,qInds)))


    multiplier = array - indexes
    multiplier = np.where(multiplier > 0, multiplier, 0)
    #print(multiplier, "\n\n\n")
    return np.where(multiplier < 1, multiplier, 1)
def GetDistances(array,indPositions,barrierElement,blanckElement,distances=["right","left","down","up"]):
    #A barreira não pode ser igual a 0******
    #Array = array com objetos
    xarray = indPositions[:, 0]  # +2# xarray [quantIndividuos,1]
    yarray = indPositions[:, 1]  # +2# yarray [quantIndividuos,1]

    modeloY = GetModel01(len(yarray), array.shape[0], yarray)#Modelo Y está correto
    modeloX = GetModel01(len(xarray), array.shape[1], xarray)#Modelo X está correto
    Ydirections, Xdirections = GetDirectionsXY(array,xarray,yarray,indPositions.shape[0])

    test1 = modeloX * Xdirections
    test2 = modeloY * Ydirections
    test3 = np.flip(Invert01Matrix(modeloX) * Xdirections, axis=1)
    test4 = np.flip(Invert01Matrix(modeloY) * Ydirections, axis=1)

    Distanciasdireita = np.abs(GetDistancesInMap(test1, barrierElement) - GetDistancesInMap(test1, blanckElement)) - 1  # OK
    Distanciasbaixo = np.abs(GetDistancesInMap(test2, barrierElement) - GetDistancesInMap(test2, blanckElement)) - 2  # OK
    Distanciasesquerda = np.abs(GetDistancesInMap(test3, barrierElement) - GetDistancesInMap(test3, blanckElement)) - 1
    Distanciascima = np.abs(GetDistancesInMap(test4, barrierElement) - GetDistancesInMap(test4, blanckElement)) - 1

    dic = {"right":Distanciasdireita,"left":Distanciasesquerda,"down":Distanciasbaixo,"up":Distanciascima}
    distancesOfPoints = np.transpose(np.array([dic[k] for k in distances]))

    return distancesOfPoints
def PositionsOfElementInArray(array,element_contains):
    positions = np.expand_dims(np.flatnonzero(np.core.defchararray.find(array, element_contains) != -1), axis=1)
    positions = np.insert(positions, 0, positions.transpose(), axis=1)
    positions[:, 1] = positions[:, 1]%array.shape[1]
    positions[:, 0] = positions[:, 0]//array.shape[0]
    return positions
def AnglesBetweensPoints(points,computeJustAngulo=None):
    #[[x1, y1], [x2, y2], [x3, y3]]
    #[[x1, y1], [x2, y2], [x3, y3]]
    #[[x1, y1], [x2, y2], [x3, y3]]
    #[[x1, y1], [x2, y2], [x3, y3]]
    #[[x1, y1], [x2, y2], [x3, y3]]
    lados = None
    first = True
    for i in range(0,3):
        lado = np.sqrt(np.sum(np.power(points[:,(i+1)%points.shape[1],:]-points[:,i,:],2),axis=1)) # sqrt(sum([x2-x1 , y2-y1]²))
        if first:
            lados = np.expand_dims(lado.copy(),axis=1)
            first = False
        else:
            lados = np.insert(lados,i,lado.copy(),axis=1)
    angulos = None
    first = True
    if computeJustAngulo==None:
        for i in range(0,3):
            #  angle = arccos((b²+c²-a²)/(2*b*c))
            angle = np.arccos(
                (np.power(lados[:,i],2)+
                np.power(lados[:,(i+1)%points.shape[1]],2)-
                np.power(lados[:,(i+2)%points.shape[1]],2))/

                (2*lados[:,i]*lados[:,(i+1)%points.shape[1]])
            )
            if first:
                angulos = np.expand_dims(angle.copy(),axis=1)
                first = False
            else:
                angulos = np.insert(angulos,i,angle.copy(),axis=1)
        return angulos
    else:
        return np.arccos(
            (np.power(lados[:, computeJustAngulo], 2) +
             np.power(lados[:, (computeJustAngulo + 1) % points.shape[1]], 2) -
             np.power(lados[:, (computeJustAngulo + 2) % points.shape[1]], 2)) /

            (2 * lados[:, computeJustAngulo] * lados[:, (computeJustAngulo + 1) % points.shape[1]])
        )
def SortPositionsByClock(positions,center,matrix,startsIn=0):
    #positions = np.array([[x1,y1],[x2,y2],[x3,y3],...])
    #print("\n\n\n\n    >>>> center: ",center)#np.array(self.matrix.shape) / 2)
    positionsSort = np.flip(positions.copy(), axis=1)
    positionsSort[:, 1] = matrix.shape[0] - positionsSort[:, 1] - 1
    #print("positions:\n",positionsSort)
    points = np.expand_dims(positionsSort.copy(), axis=1)
    points = np.insert(points, 0, np.floor(center + np.array((0, 10))), axis=1)
    points = np.insert(points, 0, np.floor(center), axis=1)
    angulos = AnglesBetweensPoints(points,computeJustAngulo=2)
    antina = 2*math.pi-angulos
    angulos = np.where(positionsSort[:,0]-center[0]>=0,angulos,antina)
    angulos = (angulos+startsIn)%(2*math.pi)
    #print("angulos: ",np.degrees(angulos))
    dicionario = {angulos[i]:positions[i] for i in range(len(angulos))}
    #print("dicionario:          ",dicionario)
    return np.array([dicionario[i] for i in np.sort(angulos).tolist()])
