import numpy as np
import random
import time

m = np.array([8, 3, 5, 2]) #masa przedmiotów
M = np.sum(m)/2 #niech maksymalna masa plecaka będzie równa połowie masy przedmiotów
p = np.array([16, 8, 9, 6]) #wartość przedmiotów

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


def generateKnapsackProblems():
    weightRow = np.array([], dtype=int)
    valueRow = np.array([], dtype=int)
    for i in range(24):
        randomWeight = random.randint(1,10)
        weightRow = np.append(weightRow, [randomWeight])
        randomValue = random.randint(1,20)
        valueRow = np.append(valueRow, [randomValue])
    return weightRow, valueRow

if __name__ == "__main__":
    weights, values = generateKnapsackProblems()
    WMAX = int(np.sum(weights)/2)
    start = time.process_time()
    knapsackProblemBruteForce(weights,values,WMAX)
    end = time.process_time()
    total = end - start
    print("{0:02f}s".format(total))
    