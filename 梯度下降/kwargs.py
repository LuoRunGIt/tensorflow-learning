# kwargs 将键值对之间转化为字典
def test(a, *args, **kwargs):
    print("a:", a)
    # print b
    # print c
    print("args:", args)

    print("kwargs:", kwargs)


test(1, 2, 3, d='4', e=5)
'''
a: 1
args: (2, 3)
kwargs: {'d': '4', 'e': 5}
'''