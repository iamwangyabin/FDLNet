



# gai


# fangfazuoyong
# 第一，自己该的network是否有同样结果
# 第二，自己设计的是否有同样结果
















# //该类中定义了四叉树创建的函数以及树中结点的属性
# //bool bNoMore： 根据该结点中被分配的特征点的数目来决定是否继续对其进行分割
# //DivisionNode()：实现如何对一个结点进行分割
# //vKeys：用来存储被分配到该结点区域内的所有特征点
# //UL, UR, BL, BR：四个点定义了一个结点的区域
# //lit:list的迭代器，遍历所有生成的节点

class ExtractorNode():
    def __init__(self, bNoMore: bool):
        self.bNoMore = bNoMore
        self.vKeys = []
        #        [x, y]
        self.UL = [0, 0]
        self.UR = [0, 0]
        self.BL = [0, 0]
        self.BR = [0, 0]

    def DivideNode(self, n1, n2, n3, n4):
        halfX = int((self.UR[0] - self.UL[0]) / 2)
        halfY = int((self.UR[1] - self.UL[1]) / 2)

        n1.UL = self.UL
        n1.UR = [self.UL[0] + halfX, self.UL[1]]
        n1.BL = [self.UL[0], self.UL[1] + halfY]
        n1.BR = [self.UL[0] + halfX, self.UL[1] + halfY]

        n2.UL = n1.UR
        n2.UR = self.UR
        n2.BL = n1.BR
        n2.BR = [self.UR[0], self.UL[1] + halfY]

        n3.UL = n1.BL
        n3.UR = n1.BR
        n3.BL = self.BL
        n3.BR = [n1.BR[0], self.BL[1]]

        n4.UL = n3.UR
        n4.UR = n2.BR
        n4.BL = n3.BR
        n4.BR = self.BR

        for i in len(self.vKeys):
            kp = self.vKeys[i]
            if kp[0]<n1.UR[0]:
                if kp[1]<n1.BR[1]:
                    n1.vKeys.append(kp)
                else:
                    n3.vKeys.append(kp)
            elif kp[1]<n1.BR[1]:
                n2.vKeys.append(kp)
            else:
                n4.vKeys.append(kp)

        if len(n1.vKeys) == 1:
            n1.bNoMore = True
        if len(n2.vKeys) == 1:
            n2.bNoMore = True
        if len(n3.vKeys) == 1:
            n3.bNoMore = True
        if len(n4.vKeys) == 1:
            n4.bNoMore = True

def DistributeOctTree(vToDistributeKeys: list, minX: int, maxX: int, minY: int, maxY: int):
    nIni = round((maxX - minX) / (maxY - minY))
    hX = (maxX - minX) / nIni
    lNodes = []
    vpIniNodes = []
    for i in range(nIni):
        ni = ExtractorNode(False)
        ni.UL = [hX * i, 0]
        ni.UR = [hX * (i + 1), 0]
        ni.BL = [ni.UL[0], maxY - minY]
        ni.BR = [ni.UR[0], maxY - minY]
        lNodes.append(ni)
        vpIniNodes.append(ni)

    for kp in vToDistributeKeys:
        vpIniNodes[kp[0] / hX].vKeys.append(kp)

    for lit in lNodes:
        if len(lit.vKeys) == 1:
            lit.bNoMore = True
        elif len(lit.vKeys) == 0:
            lNodes.remove(lit)  # ?????????????????????????
        else:
            pass

    bFinish = False
