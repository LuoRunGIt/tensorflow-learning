class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        #第一反应是分割
        end=len(s)-1
        i=0
        while s[end]==' ':
            end=end-1
        while s[end]!=' 'and end>=0:
            end=end-1
            i=i+1
        return i
# strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）
        #return len(s.strip().split()[-1])


#数字加1
digits=[1,1,2]
#digits.insert(0,1)
digit = [0 for i in range(len(digits)+1)]

digit[0]=1
print(digit)


def plusOne( digits: list[int]) -> list[int]:
    right = len(digits)
    while right > 0:
        if digits[right - 1] != 9:
            digits[right - 1] += 1
            return digits

        else:
            digits[right - 1] = 0
            right -= 1
    else:
        digit = [0 for i in range(len(digits) + 1)]
        digit[0] = 1
        return digit
k=plusOne([9,9])
#只有99，999，9999，这样的数才会导致数组的扩展
print(k)