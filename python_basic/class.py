# -*- coding: utf-8 -*-
# @Author: tjk
# @Date:   2020-03-11 11:25:06
# @Last Modified by:   tjk
# @Last Modified time: 2020-03-11 16:33:42




###继承和多态
class Animal(object):
	def run(self):
		print("Animal can run")
	def __len__(self):
		return 100
    
    	

class Dog(Animal):
	def run(self):
		print("Dog run fast")
 
class Cat(Animal):
	pass


dog = Dog()
cat = Cat()
dog.run()
cat.run
print(len(dog))

class Screen(object):
	def __init__(self,name):
		self.name = name
	def __str__(self):
		return 'name:%s'% self.name
	@property
	def height(self):
		return self._height
	@height.setter
	def height(self,value):
		self._height = value
    
	@property
	def width(self):
		return self._width
	@width.setter
	def width(self,value):
		self._width = value

	@property
	def resolution(self):
		return self._width * self._height
# 测试:
s = Screen('fdf')
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')

print(s)