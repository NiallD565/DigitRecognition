import os

os.chdir('')

i = 0

print(os.getcwd())

for f in os.listdir():	
	os.rename(f + i.str())
	
	i++
	