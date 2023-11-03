import matplotlib.pyplot as plt
import numpy as np

with open('D:\Work\DP\DT_Rail_object_detection\models//log_300_0.1_adadelta.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    values = []
    for line in lines:
        values.append(str(line)[-6:-1])

test_data = []
train_data = []

for i,value in enumerate(values):
    if i % 2 == 0:
        train_data.append(float(value))
    else:
        test_data.append(float(value))

x = np.arange(0,len(lines)/2)

plt.style.use('bmh')
plt.plot(x, train_data)
plt.title('Train loss')
#plt.xticks(x)
plt.show()

plt.plot(x, test_data)
plt.title('Test loss')
#plt.xticks(x)
plt.show()
