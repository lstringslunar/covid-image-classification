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

accuracy = []
for i in range(30):
    line = lines[index]
    line = line.split()
    # print(type(line[2]))
    accuracy.append(float(line[2]))
    index += 5

print(len(accuracy))
print(accuracy)
