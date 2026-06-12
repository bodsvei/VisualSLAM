import re

with open('demo_test.log', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'Frame 1850' in line:
        start = i
    if 'Frame 1950' in line:
        end = i
        break

print("".join(lines[start:end+1]))

