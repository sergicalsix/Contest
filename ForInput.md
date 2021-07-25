# 1行
## 1文字
`s = input()`

`s = int(input())`

## 複数文字
`s = input.split()`

`a, b = map(int, input().split())` 

`l = list(map(int, input().split()))`

# 複数行
## (N,1)
`N, M = map(int, input().split())`

`A = [int(input()) for _ in range(M)]`

## ３変数
```N = int(input())
t = [0] * N
x = [0] * N
y = [0] * N
for i in range(N):
    t[i], x[i], y[i] = map(int, input().split())```




### 参考:
https://qiita.com/jamjamjam/items/e066b8c7bc85487c0785
