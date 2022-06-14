path = 'x64-30ep/ensemble/2.txt'

file = open(path, 'r', encoding='utf-8')
text = file.read()
file.close()
text = text.split('\n')
lines = []
for line in text:
    lines.append(line)

print(len(lines))
index = 2

f1_score = []
for i in range(30):
    accuracy = float(lines[index].split()[2])
    precision = float(lines[index + 1].split()[2])
    recall = float(lines[index + 2].split()[2])
    # print(type(line[2]))
    f1 = 0
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    f1_score.append(f1)
    index += 5

print(len(f1_score))
print(f1_score)
