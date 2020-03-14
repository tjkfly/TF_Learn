# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-10 14:58:02
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-10 18:41:09
def invest(amount , rate, time=1):
	total = amount
	print("principle amount:",amount)
	rate = rate /100.
	for i in range(1,time+1):
		total += total * rate 
		print("year",i,': $',total)



if __name__=='__main__':
	invest(100,5,8)
	dict={'Name': 23, 'Age': 27}
	dict['Name'] = dict.get('Name',0)+1
	print(dict)