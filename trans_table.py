#!/usr/bin/env python3

s = '''Target: 359.0 -> 317.0
Target: 450.0 -> 467.3
Target: 475.0 -> 354.6
Target: 650.0 -> 587.3
Target: 685.0 -> 890.9
Target: 880.0 -> 1105.1
Target: 395.84 -> 334.9
Target: 350.0 -> 277.4
Target: 750.0 -> 860.6
Target: 295.0 -> 192.0'''

ts = []
ys = []


for line in s.split('\n'):
    words = line.split()

    t = float(words[1])
    y = float(words[3])

    ts.append(t)
    ys.append(y)


for t in ts:
    print(t, end=" & ")

print()

for y in ys:
    print(y, end=" & ")

print()