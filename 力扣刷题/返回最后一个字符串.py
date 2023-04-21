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

        #return len(s.strip().split()[-1])