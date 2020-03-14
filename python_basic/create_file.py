# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-10 14:42:36
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-10 15:28:41
# -*- coding: utf-8 -*-

path = '/home/tjk/project/python_basic/'

for i in range(1,11):
	full_path = path + str(i)+'.txt'
	file = open(full_path,'w')
	msg = str(i)
	file.write(msg)
	file.close()
	