X = input()
a = int(X[0])
b = int(X[1])
c = int(X[2])
d = int(X[3])

if (a == b) and (c == d):
    if (b == c):
        print("Weak")
        exit()
if (a+1) % 10 == b  :
    if (b+1) %10 == c :
        if (c+1) % 10 == d  :
            print("Weak")
            exit()

print("Strong")
