import numpy as np
import random
import time



def knapsackProblemBruteForce(m, p, Wmax):
    n = len(m)
    maxValue = 0
    for i in range(2**n):
        currentWeight = 0
        currentValue = 0
        for j in range(n):
            if (i & (2**(n-1-j))):
                if currentWeight + m[j] <= Wmax:
                    currentWeight += m[j]
                    currentValue += p[j]

        maxValue = max(currentValue, maxValue)
    return maxValue


def sortElements(m, p):
    n = len(m)
    for i in range(n):
        for j in range(n-1):
            if p[j]/m[j] < p[j+1]/m[j+1]:
                tempValue = p[j+1]
                tempWeight = m[j+1]
                p[j+1] = p[j]
                m[j+1] = m[j]
                p[j] = tempValue
                m[j] = tempWeight
    return (m,p)


def packElemetsUsingPMRatio(m,p, Wmax):
    m, p = sortElements(m, p)
    n = len(m)
    currentWeight = 0
    maxValue = 0
    for i in range(n):
        if currentWeight + m[i] <= Wmax:
            currentWeight += m[i]
            maxValue += p[i]
        else:
            break
    return maxValue



def generateKnapsackProblems(amount):
    weightRow = np.array([], dtype=int)
    valueRow = np.array([], dtype=int)
    for i in range(amount):
        randomWeight = random.randint(1,10)
        weightRow = np.append(weightRow, [randomWeight])
        randomValue = random.randint(1,20)
        valueRow = np.append(valueRow, [randomValue])
    return weightRow, valueRow



def main():
    m = np.array([8, 3, 5, 2]) #masa przedmiotów
    M = np.sum(m)/2 #niech maksymalna masa plecaka będzie równa połowie masy przedmiotów
    p = np.array([16, 8, 9, 6]) #wartość przedmiotów
    amount = 16
    weights, values = generateKnapsackProblems(amount)
    Wmax = int(np.sum(weights)/2)
    # start = time.process_time()
    packElemetsUsingPMRatio(weights,values,Wmax)
    # end = time.process_time()
    # total = end - start
    # print(f"{total:.8f}s \n")
    # start = time.process_time()
    knapsackProblemBruteForce(weights,values,Wmax)
    # end = time.process_time()
    # total = end - start
    # print(f"{total:.8f}s \n")

if __name__ == "__main__":
    main()
