import pandas as pd

dataList = pd.read_csv("data/data.csv")
lenN = dataList.columns.size
NumFall = [0] * lenN
ValueFall = []
for j in range(lenN):
    ValueSingle = []
    for i in range(len(dataList)):
        if dataList.iloc[i,j] not in ValueSingle:
            ValueSingle.append(dataList.iloc[i,j])
    ValueFall.append(ValueSingle)
    NumFall[j] = len(ValueSingle)   
# print("ValueFall: ",ValueFall)
# print("NumFall: ",NumFall)

def count(list,data):
    number = 0
    for i in range(len(data)):
        isIn = True
        for j in range(len(list)):
            if list[j] not in data.iloc[i].to_list():
                isIn = False
                break
        if isIn:
            number += 1
    return number
def BUC(tempList, n, curN, min_sup = 3):
    if curN == n:
        return
    for i in range(NumFall[curN]):
        tempList.append(ValueFall[curN][i])
        if count(tempList, dataList) >= min_sup:
            if len(tempList) <= 2:
             print("%s: %d" % (tempList, count(tempList, dataList)))
             BUC(tempList, n, curN+1)
        tempList.pop()
    BUC(tempList, n, curN+1)
BUC([], 6, 0)