import numpy as np
N, M = map(int, input().split())
A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))
big = 100_000_000_000

#M回目の計算
if N - M >= 0:
    ans = 10000000
    B = np.append(B, [big]* (N - M))
    for i in range(M):
        min_ = np.amin(A - B)
        tmp = B[1:]
        B = np.append(tmp,B[0])

        if ans  > min_:
            ans = min_

elif M - N > 0:
    ans = 10000000
    A = np.append(A, [big]*(M - N))
    for i in range(N):
        min_ = np.amin(A - B)
        tmp = A[1:]
        A = np.append(tmp,A[0])

        if ans  > min_:
            ans = min_

print(ans)
