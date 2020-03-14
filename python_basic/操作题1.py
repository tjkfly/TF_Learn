# !usr/bin/env python
# -*- coding:utf-8 -*-


# # 操作题1.1
# N = input('请输入0-100内的整数:')  # N取值范围是0—100，整数
# n = int(N)
# e = " " * (3-len(N))
# print(f"{e}{n}%@{ '=' * (n//5)}")


# # 操作题1.2
# s = '学而时习之，可以为师矣！'
# n = s.count('，') + s.count('！')
# m = len(s) - n
# print(f'字符数为{m}，标点符号数为{n}')


# # 操作题1.3
# N = input("请输入一个整数: ")
# n = int(N)
# if n != 0:
#     s = 0
#     for i in range(n,n+100,2):
#         s += i
# else:
#     for i in range(n+1,n+100,2):
#         s += i
# print(f'您输入的整数 N 到 N+100 之间所有奇数的数值和为{s}')


# # 操作题1.4
# import turtle as t
# for i in range(0,6):
#     t.fd(200)
#     t.left(60)


# # 操作题1.5
# def getInput():
#     while True:
#         try:
#             N = input('请输入一个正整数：')
#             n = int(N)
#             print(n)
#             break
#         except:
#             input('请输入一个正整数：')
#
# getInput()


# 操作题1.6.1
import jieba

count1 = {}
count2 = {}
ls1 = []
ls2 = []

# 读取文本，构造出字典
with open('天龙八部.txt', 'r', encoding='utf-8') as f:   # 打开文本
    words = f.read()                                        # 读取
    words_list = jieba.lcut(words)                          # 分词
    for word in words_list:
        if len(word) == 1:
            count1[word] = count1.get(word, 0) + 1          # 汉字创造词典
        else:
            count2[word] = count2.get(word, 0) + 1          # 词语创造词典

# 将字典按照值从大到小排序
# 该段程序非必要，我为了查漏补缺知识点加上的
dict2list_1 = sorted(count1.items(),key = lambda x:x[1],reverse = True)
count1 = dict(dict2list_1)
dict2list_2 = sorted(count2.items(),key = lambda x:x[1],reverse = True)
count2 = dict(dict2list_2)

# 将字典转换为指定输入形式的列表
for key,value in count1.items():
    ls1.append(f'{key}:{value}')
for key,value in count2.items():
    ls2.append(f'{key}:{value}')

# 创造两个csv文件并打开写入内容
with open('天龙八部-汉字统计.csv', 'w') as f:
    for i in ls1:
        if ('\n' or ' ') not in i:
            f.write(i + ', ')
    print('汉字统计导入成功')

with open('天龙八部-词语统计.csv', 'w') as f:
    for i in ls2:
        f.write(i + ', ')
    print('词语统计导入成功')
