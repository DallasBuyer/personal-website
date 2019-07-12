---
title: Search Algorithms based on Linear List
date: 2019-06-16 15:58:34
tags: data structure, algorithm
categories: data structures and algorithms
---

### 顺序查找
顺序查找一般是针对**无序表**的线性查找，遍历所有元素。最好的情况是第一个位置就找到了，这时候算法的时间复杂度为 $O(1)$，最坏的情况是遍历到最后一个位置才找到，这时候时间复杂度为 $O(n)$，所以平均查找次数为 $(n+1)/2$。最终时间复杂度为 $O(n)​$。

```python
class Find:
    def seqSearch(self, lists, key):
        length = len(lists)
        if length == 0:
            return False
        for i in range(length):
            if lists[i] == key:
                return i
        else:
            return False


if __name__ == "__main__":
    lists = [1, 5, 8, 123, 22, 54, 7, 99, 300, 222]
    find = Find()
    results = find.seqSearch(lists, 123)
    print(results)
```
## <!--more-->


### 二分查找

二分查找又叫折半查找，优点是比较次数少，查找速度快，平均性能好。缺点是要求待查表为**有序表**，且插入删除操作困难。所以折半查找适用于不经常变动而查找频繁的有序列表。首先，假设表中元素是按升序排列，将表中间位置记录的关键字与查找关键字比较，如果两者相等，则查找成功；否则利用中间位置记录将表分成前、后两个子表，如果中间位置记录的关键字大于查找关键字，则进一步查找前一子表，否则进一步查找后一子表。重复以上过程，直到找到满足条件的记录，使查找成功，或直到子表不存在为止，此时查找不成功。最好的情况时间复杂度为 $O(1)$，最坏的情况时间复杂度为 $O(logn)​$。下面是二分查找的两种Python实现，一种递归一种非递归。

```python
class Find:
    def binarySearch(self, lists, key):
        length = len(lists)
        if length == 0:
            return False
        begin = 0
        end = length - 1
        while begin <= end:
            mid = (begin + end) // 2
            if lists[mid] > key:
                end = mid - 1
            elif lists[mid] < key:
                begin = mid + 1
            else:
                return True
        return False
    
    def binaryRecursiveSearch(self, lists, key):
        length = len(lists)
        if length == 0:
            return False
        mid = length // 2
        if lists[mid] > key:
            return self.binaryRecursiveSearch(lists[0:mid], key)
        elif lists[mid] < key:
            return self.binaryRecursiveSearch(lists[mid+1:], key)
        else:
            return True

if __name__ == "__main__":
    lists = [1, 5, 8, 123, 22, 54, 7, 99, 300, 222]
    find = Find()
    results = find.binaryRecursiveSearch(lists, 54)
    print(results)
```



### 插入查找





### 斐波那契查找





### 散列表查找(哈希表)



