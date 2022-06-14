from datetime import datetime

import matplotlib.pyplot as plt

acc_file = open('f1-score-ensemble.txt', 'r')
text = acc_file.read()
text = text.split('\n')
acc = []
for i in range(len(text)):
    acc.append([float(f) for f in text[i].split(',')])
    print(len(acc[i]))

model_file = open('models-ensemble.txt', 'r')
text = model_file.read()
models = text.split('\n')

plt.title('Ensemble F1-score')
plt.xlabel('epochs')
plt.ylabel('f1-score')

ep = range(0, 31)
color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i in range(len(models)):
    plt.plot(ep, acc[i], color[i], label=models[i])
plt.xlim([1, 30])
plt.ylim([0, 1])
plt.legend(loc='lower right')
now = datetime.now().strftime("%m%d-%H%M%S")

plt.savefig('ensemble-f1score-plot-' + now + '.png')
plt.show()
