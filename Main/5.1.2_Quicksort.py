import random

# Set seed.
random.seed(a=100)


def swapPositions(list, pos1, pos2): 
    if pos1 != pos2:
        # popping both the elements from list 
        first_ele = list.pop(pos1)    
        second_ele = list.pop(pos2-1) 
     
        # inserting in each others positions 
        list.insert(pos1, second_ele)   
        list.insert(pos2, first_ele)   
      
    return list

def quicksort(A, lo, hi):
    if lo < hi:
        p, A = partition(A, lo, hi)
        A = quicksort(A, lo, p - 1)
        A = quicksort(A, p + 1, hi)
    return A

def partition(A, lo, hi):
    pivot = A[hi]
    i = lo
    for j in range(lo, hi):
        if A[j] < pivot:
            A = swapPositions(A, i, j)
            i = i + 1
    A = swapPositions(A, i, hi)
    return i, A

short_list = list(random.sample(range(1000000), 10))
print(short_list)
print(quicksort(short_list, 0, 9))