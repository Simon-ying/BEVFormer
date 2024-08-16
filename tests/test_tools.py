import time

a = 8
b = 3
start = time.time()
for i in range(100):
    c = int(a/b)
end1 = time.time()
for i in range(100):
    d = a // b
end2 = time.time()

print(f"int(a/b): {end1-start} s, a//b: {end2-end1} s")