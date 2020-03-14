# !usr/bin/env python
# -*- coding:utf-8 -*-

# 习题讲解
# 参考代码
# 1
# fi = open("天龙八部.txt", "r", encoding='utf-8')
# fo = open("天龙八部-汉字统计.txt", "w", encoding='utf-8')
# txt = fi.read()
# d = {}
# for c in txt:
#     d[c] = d.get(c, 0) + 1
# del d[' ']
# del d['\n']
# ls = []
# for key in d:
#     ls.append("{}:{}".format(key, d[key]))
# fo.write(",".join(ls))
# fi.close()
# fo.close()

# 参考代码
# 2

import jieba

# fi = open("天龙八部.txt", "r", encoding='utf-8')
# fo = open("天龙八部-词语统计.txt", "w", encoding='utf-8')
# txt = fi.read()
# words = jieba.lcut(txt)
# d = {}
# for w in words:
#     d[w] = d.get(w, 0) + 1
# del d[' ']
# del d['\n']
# ls = []
# for key in d:
#     ls.append("{}:{}".format(key, d[key]))
# fo.write(",".join(ls))
# fi.close()
# fo.close()
