class Solution:
    def addBinary(self, a: str, b: str) -> str:
        c = int(a, 2) + int(b, 2)
        return str(bin(c))[2:]
a="100"
b="100"
c = int(a, 2) + int(b, 2)
print(c)
print(bin(c))