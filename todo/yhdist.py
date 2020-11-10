import os
total = ''
for filename in os.listdir():
	if filename[:6] == 'output':
		with open(filename,'r') as data:
			total+=data.read()
data =  total.split('\n')[:-1]
print(data)
index1=0
while index1 < len(data):
	item = data[index1]
	index2 = index1+1
	amountsize = item.index('x')
	itemamount = int(item[0:amountsize])
	while index2 < len(data):
		item2 = data[index2]
		amountsize2 = item.index('x')
		itemamount2 = int(item2[0:amountsize])
		if item[amountsize:].lower() == item2[amountsize2:].lower():
			data[index1] = str(itemamount+itemamount2)+item[amountsize:]
			del data[index2]
		index2+=1
	index1+=1
with open('combined.txt','a') as f:
	f.write('\nResult:\n')
	for line in data:
		f.write(line+'\n')