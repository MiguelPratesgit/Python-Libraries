import numpy as np
import cv2
import math


def SobreporComMascara(background, foreground, mask):
    # mask = np.zeros(self.image.shape[:2], dtype="uint8")
    # cv2.circle(mask, clickable.area[0], clickable.area[1], 255, -1)  # clickable.color
    #https://medium.com/featurepreneur/performing-bitwise-operations-on-images-using-opencv-6fd5c3cd72a7

    mask_inv = cv2.bitwise_not(mask)

    fore = cv2.bitwise_and(foreground, foreground, mask=mask)
    back = cv2.bitwise_and(background, background, mask=mask_inv)
    return cv2.add(back, fore)
#color = [blue,green,red]
AREA_CIRCULAR=0     #area = [p,r]
AREA_RECTANGLE=1    #area = [p1,p2]
AREA_POLYGON=2      #area = [p1,p2,p3...] *Usa os polygons

class ClickableObject():
    def __init__(self,area,areatype,function,
                 image=None,
                 color=None,
                 order=0,hide=False,nome=None,tag=None,attributes=None):
        self.order = order
        self.nome = nome
        self.tag = tag
        self.attributes = attributes
        self.areatype = None
        self.area = None
        self.function = function
        self.hide=hide
        self.Atualize(areatype=areatype,area=area,image=image,color=color,)
    def Atualize(self,areatype=None,area=None,image=None,color=None):
        if areatype!=None:
            self.areatype=areatype
        if area!=None:
            self.area=area
            if self.areatype == AREA_RECTANGLE:
                self.areatype = AREA_POLYGON
                self.area = [[area[0][0], area[0][1]],# [0, 50],
                             [area[0][0], area[1][1]],# [p1[1],p2[0]],
                             [area[1][0], area[1][1]],# [50, 600],
                             [area[1][0], area[0][1]]]# [p1[0],p2[1]]
            if self.areatype == AREA_POLYGON:
                self.area= np.flip(np.array(self.area),axis=1)
        if not image is None:
            self.image=image
        if color!=None:
            self.color=color

        #print("area type: ",self.areatype)
        if self.areatype == AREA_CIRCULAR:
            #print("area circular")
            self.image_InicialPoint = np.array(self.area[0]) - np.array((self.area[1], self.area[1]))#mudar??
            self.image_HeightWidth = 2 * np.array((self.area[1], self.area[1]))

            self.image_InicialPoint = np.flip(self.image_InicialPoint)
            self.image_HeightWidth = np.flip(self.image_HeightWidth)

            #if self.color!=None:
            #    self.image = np.full((self.image_HeightWidth[1], self.image_HeightWidth[0], 3), color, dtype='uint8')
            #self.image = cv2.resize(self.image, self.image_HeightWidth)

        if self.areatype==AREA_POLYGON:
            #print("    >>Polygon")
            aux = np.array(area)
            self.image_InicialPoint = np.array([np.min(aux[:,0]),np.min(aux[:,1])])
            self.image_HeightWidth = np.array([np.max(aux[:,0]),np.max(aux[:,1])])-self.image_InicialPoint

            self.image_InicialPoint = np.flip(self.image_InicialPoint)
            self.image_HeightWidth = np.flip(self.image_HeightWidth)



        if color!=None:
            self.image = np.full((self.image_HeightWidth[1], self.image_HeightWidth[0], 3), color, dtype='uint8')
        #print("               self.image_InicialPoint",self.image_InicialPoint,"             self.image_HeightWidth:",self.image_HeightWidth)
        self.image = cv2.resize(self.image, self.image_HeightWidth)
        self.Recoverlastatualizedimage()
        return self
    def CreateMask(self,backgroundShape):
        if self.areatype == AREA_CIRCULAR:
            #print("maskara circular")
            mask = np.zeros(backgroundShape, dtype="uint8")
            cv2.circle(mask, np.flip(self.area[0]), self.area[1], 255, -1)  # clickable.color

        if self.areatype == AREA_POLYGON:
            mask = cv2.fillPoly(np.full((backgroundShape[0], backgroundShape[1], 3), [0, 0, 0], dtype='uint8')
                                , [np.array(self.area)], [255, 255, 255])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask
    def SobreporEmBackground(self,background,mask,color_exclude=None):

        icon = np.full((background.shape[0], background.shape[1], 3), [255, 255, 255], dtype='uint8')
        #funciona com polygons
        icon[self.image_InicialPoint[1]:self.image_InicialPoint[1] + self.image_HeightWidth[1],
             self.image_InicialPoint[0]:self.image_InicialPoint[0] + self.image_HeightWidth[0]] = self.Showimage

        if color_exclude!=None:
            threshold = cv2.inRange(icon,np.array([255,255,255]),
                                    np.array([255,255,255]))
            maskara = cv2.bitwise_not(threshold)
        else:
            maskara = mask
        return SobreporComMascara(background, icon,maskara)
    def Recoverlastatualizedimage(self):
        self.Showimage = self.image.copy()
    def Getcenter(self):
        if self.areatype==AREA_CIRCULAR:
            return np.array(self.area[0])
        else:#areatype==AREA_POLYGON:
            return np.ndarray.astype(np.mean(np.array(self.area),axis=0),dtype=int)
    def Move(self,avancar):#avancar=[yunits,xunits]
        if self.areatype==AREA_CIRCULAR:
            self.area[0][0] += avancar[0]
            self.area[0][1] += avancar[1]
        else:#self.areatype==AREA_POLYGON:
            self.area = np.array(self.area)
            self.area[:, 0] += avancar[0]
            self.area[:, 1] += avancar[1]
            self.area = self.area.tolist()
        print("\n\n\n\n\nobjeto movido")
        self.Atualize(area=self.area)

    def MoveTo(self,pos):#pos=[yir,xir]
        self.Move(np.array(pos)-self.Getcenter())
        pass



DISPLAY_RING = 0    #displayarea [p,r]
DISPLAY_BOX = 1     #displayarea [[w,h],[realW,realH]]
class ClickableScreen():
    def __init__(self,screenDim,screenTouchFunc=lambda x,y,evt,i:print("ouch (°J°)"),namedWindow="clickableScreen",clickEvent=[cv2.EVENT_LBUTTONDOWN],sobreposicao='first',backgroundImage=None):
        #sobreposicao = 'first','all','first+background'
        self.screenDim = screenDim
        self.clickables = [ClickableObject([[0,0],screenDim],AREA_RECTANGLE,screenTouchFunc,color=[255,255,255],nome="screen",image=backgroundImage,tag="Screen")]
        print("screen area:\n",self.clickables[0].area)
        self.windowname = namedWindow
        self.clickEvent = clickEvent
        self.orderCounter = 0
        self.sobreposicao = sobreposicao

        #cv2.imshow("maskara background",self.clickables[0].CreateMask(self.screenDim))
        #cv2.waitKey()

        self.masks = np.expand_dims(self.clickables[0].CreateMask(self.screenDim),axis=0)
        print("\n\n\n\n---------------------0-0-0-0-0-0-0-0----------")
    def Atualize(self):
        #print("masks antes: ",self.masks.shape)
        self.masks = np.expand_dims(self.clickables[0].CreateMask(self.screenDim), axis=0)
        for clickable in self.clickables[1:]:
            mask = clickable.CreateMask(self.screenDim)
            self.masks = np.insert(self.masks, len(self.masks), mask, 0)
        #print("masks depois: ", self.masks.shape)
    def AddButton(self,clickable):
        self.orderCounter+=1
        clickable.order = self.orderCounter
        self.clickables.append(clickable)
        mask = clickable.CreateMask(self.screenDim)
        self.masks = np.insert(self.masks,len(self.masks),mask,0)
    def CreateImage(self):
        def click(event, x, y, flags, param):
            if event in self.clickEvent:
                arrayClicados = np.where(self.masks[:,y,x]==255)[0]
                #print("arrayClicados: ",arrayClicados)
                dropHides = []
                if len(arrayClicados)>0:
                    for i in arrayClicados:
                        if self.clickables[i].hide:
                            dropHides.extend(np.where(arrayClicados==i)[0])
                    if len(dropHides)>0:
                        arrayClicados = np.delete(arrayClicados,dropHides)
                    if self.sobreposicao=='first':
                        arrayClicados = [np.max(arrayClicados)]
                    if self.sobreposicao=='first+background':
                        if len(arrayClicados)>1:
                            arrayClicados = [arrayClicados[0],np.max(arrayClicados)]
                        else:
                            arrayClicados = [np.max(arrayClicados)]
                    for i in arrayClicados:
                        print("\n\n",self.clickables[i].nome,"executou a função")
                        self.clickables[i].function(x,y,event,i)

        cv2.namedWindow(self.windowname)
        cv2.setMouseCallback(self.windowname, click)
        #ShowScreenLoop))

        while True:
            #print(".___.")
            self.image = np.full((self.screenDim[0], self.screenDim[1], 3), [255, 255, 255],
                                 dtype=np.uint8)  # Coloca o background
            for clickable in self.clickables:
                #if clickable.areatype==AREA_CIRCULAR:
                    #print("       point: ",clickable.image_InicialPoint,"    hw:",clickable.image_HeightWidth,"    area: ",clickable.area)
                if not clickable.hide:
                    #cv2.imshow("clickableimage",clickable.image)
                    self.image = clickable.SobreporEmBackground(
                        self.image,clickable.CreateMask(self.image.shape[:2]))
            cv2.imshow(self.windowname, self.image)
            key = cv2.waitKey(1)
            if key==32:
                for mask in self.masks:
                    cv2.imshow("máscara", mask)
                    cv2.waitKey(2000)
                #print("len(self.clickables): ",len(self.clickables))
    def CreateNomeList(self):
        lista = []
        for clickable in self.clickables:
            lista.append(clickable.nome)
        return lista
    def CountTags(self,tag):
        if tag==None:
            return len(self.clickables)
        else:
            i = 0
            for clickable in self.clickables:
                if clickable.tag==tag:
                    i+=1
            return i
    def SetButtonsInDisplay(self, display_type, displayarea, angleAdder=0, useTag=None):
        if display_type == DISPLAY_RING:#displayarea [p,r]
            angleDiv = 2 * math.pi/self.CountTags(useTag)
            auxAngle = angleAdder
            for clickable in self.clickables:
                if useTag == clickable.tag and useTag != None:
                    pos = [int(displayarea[1] * math.sin(auxAngle)),# - clickable.image.shape[0]/2
                           int(displayarea[1] * math.cos(auxAngle))]
                    pos = np.flip(np.array(displayarea[0]))+np.array(pos)
                    clickable.MoveTo(pos)
                    auxAngle += angleDiv
        if display_type == DISPLAY_BOX:#displayarea [[w,h],[realW,realH]]
            width_height = np.array(displayarea[1])/np.array(displayarea[0])
            for x in displayarea[0]:
                for y in displayarea[1]:
                    pos = np.array([[0, 0],
                                    [0, self.tileDim],
                                    [self.tileDim, self.tileDim],
                                    [self.tileDim, 0]])
                    pos[:, 0] += y * width_height[0]
                    pos[:, 1] += x * width_height[1]
        self.Atualize()