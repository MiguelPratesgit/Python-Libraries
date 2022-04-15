import numpy as np
import UsefulMethods.Useful as Useful
PROPAG_DIVIDEAREAS = 0
PROPAG_EXPANDLAYERS = 1
class Propagation():#UsefulMethods.Propagation(supertabarray).areas
    def __init__(self,matrix,dirs=[(0,1),(0,-1),(1,0),(-1,0)],searchingFor=[],propagType=PROPAG_DIVIDEAREAS,
                 KeepJustBorder=False):
        self.matrix = matrix
        exclude = []

        if propagType==PROPAG_DIVIDEAREAS:
            self.areas = []
            for x in range(0, len(self.matrix)):
                for y in range(0, len(self.matrix[0])):
                    #print("exclude: ",[x,y],"      ",exclude)
                    if not [x,y] in exclude:
                        itemArea = self.matrix[x, y]
                        if searchingFor==[]:
                            area = self.Propagate([x,y],[itemArea])
                            exclude.extend(area)
                            self.areas.append((itemArea, area))
                        elif self.matrix[x][y] in searchingFor:
                            area = self.Propagate([x, y], searchingFor)
                            exclude.extend(area)
                            self.areas.append((searchingFor,area))
        if propagType==PROPAG_EXPANDLAYERS:
            positions = []
            for i in searchingFor:
                positions.extend(Useful.PositionsOfElementInArray(self.matrix,i))
            positions = np.array(positions)
            positions = np.resize(positions,(len(dirs),len(positions),2))
            initialPositions = positions.copy()[0]
            for i in range(0,len(positions)):
                positions[i,:,:] += dirs[i]
            positions = np.reshape(positions, (1,-1, 2))
            positions = np.unique(positions,axis=0)[0]
            toDrop = positions<0
            for enum in enumerate(self.matrix.shape):
                toDrop[:,enum[0]] += positions[:,enum[0]]>=enum[1]
            toDrop = toDrop[:,0]+toDrop[:,1]
            positions = np.delete(positions,np.where(toDrop),axis=0)
            if KeepJustBorder:
                for i in initialPositions:
                    delete = np.sum(positions!=i,axis=1)
                    positions = positions[delete!=0]
            self.positions=positions
    def Propagate(self, pos, itemsSearched, dirs=[[0,1],[1,0],[0,-1],[-1,0]], NoneElement=0):
        self.taked = [pos]
        matrixCopy = self.matrix.copy()
        area = [pos]
        NoneElementi = NoneElement
        while True:
            if not NoneElementi in itemsSearched:
                break
            NoneElementi+=1
        while len(self.taked)>0:
            newTakes = []
            for take in self.taked:
                matrixCopy[take[0],take[1]] = NoneElementi
                #print("BATATA:", take)
                for dir in dirs:
                    newpos = np.array(take) + np.array(dir)

                    if newpos[0] >= 0 and newpos[1] >= 0 and newpos[0] < matrixCopy.shape[0] and newpos[1] < matrixCopy.shape[1]:
                        if matrixCopy[newpos[0],newpos[1]] in itemsSearched:
                            #print("new pos: ",newpos,"\n")
                            newTakes.append(newpos)
                            area.append(newpos.tolist())
            self.taked = newTakes
        return np.unique(np.array(area), axis=0).tolist()



