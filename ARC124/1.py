N, K = map(int, input().split())
#if (N - K) ==1:
#    print(1)
c = [0] * K
k = [0] * K
A = [0] * N #1 < x < K
#書き込み

for i in range(K):
    c[i], k[i] = input().split()
    k[i] = int(k[i])
    if c[i] == 'L':
        A[k[i] - 1] = i + 1

    else:
        A[N - k[i]] = i + 1
print(A)
#数え上げ
magic = 998244353
ans: int = 1
for i in range(1,N+1):
    prob = K
    if A[i-1] == 0:
        for j in range(K):
            if c[j] == 'L' and A[k[j]] > i:
                prob -= 1
            elif c[j] == 'R' and A[N-k[j]] < i:
                prob -= 1
        ans *= prob

print(ans % magic)
