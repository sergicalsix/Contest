a, b = map(int, input().split())
if (a > 0) and (b == 0):
    print("Gold")
elif (a == 0) and (0 < b):
    print("Silver")
else:
    print("Alloy")
