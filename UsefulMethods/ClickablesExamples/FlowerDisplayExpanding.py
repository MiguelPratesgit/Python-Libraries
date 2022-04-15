from UsefulMethods.Clickables import *
import math
import numpy as np

dim = (700,700)#300 de altura e 700 de comprimento (retângulo horizontal/deitado)
background = np.full((dim[0],dim[1],3),[255,255,255],dtype='uint8')
def Reposicione(x,y,evt,i):
    distanceToCenter = 2*raio*math.sin((math.pi-2*math.pi/quantBolas)/2) / math.sin(2*math.pi/quantBolas)
    clickScreen.SetButtonsInDisplay(DISPLAY_RING,[[x,y],distanceToCenter],useTag="bola",angleAdder=math.radians(45))
    AdicionarBola()
clickScreen = ClickableScreen(dim,screenTouchFunc=Reposicione)
bolas = []
raio = 25
quantBolas=0
def AdicionarBola():
    global quantBolas
    bola = ClickableObject([[raio, raio], raio],
                           AREA_CIRCULAR,
                           lambda x, y, evt, i: print("bola clicada :)"),
                           color=[0, 0, quantBolas*10], tag="bola", hide=False)
    clickScreen.AddButton(bola)
    quantBolas+=1
AdicionarBola()
AdicionarBola()



import threading
threading.Thread(target=lambda: clickScreen.CreateImage()).start()