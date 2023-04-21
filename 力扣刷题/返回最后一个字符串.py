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