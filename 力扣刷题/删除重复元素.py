nums = [1, 2, 2, 3, 3, 4]
print(nums[-1])
print(type(nums))
for i in (nums):
    print(i)

# 不含第一个数
for i in (nums[1:]):
    print(i)

# 不含最有一个数
for i in (nums[:len(nums) - 1]):
    print(i)

# 注意python始终是左闭右开
for i in range(len(nums) - 1):
    print(nums[i])

for i in range(8, 12):
    if i == 10:
        print("break")
        break
    print(i)


# 基础写法
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        max = nums[-1]
        print(max)
        b = 0
        for i in range(len(nums)):
            if nums[i] == max:
                b = i + 1
                # print(b)
                break
            else:
                if nums[i] >= nums[i + 1]:
                    for j in range(i + 1, len(nums)):
                        if nums[j] > nums[i]:
                            b = nums[i + 1]
                            nums[i + 1] = nums[j]
                            nums[j] = b
                            break
        print(b)
        return b

#快慢指针写法
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        left=0
        for right in range(1,len(nums)):
            if nums[left]!=nums[right]:
                nums[left+1]=nums[right]
                left=left+1
        return left+1