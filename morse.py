import sys

## 空白のないモールス信号を翻訳する問題
def main(lines):
    s:str = lines[0]
    ## internet から流用
    ## 特徴として1から4文字の長さ
    morse_dict = {"A": ".－","B": "－...","C": "－.－.","D": "－..","E": ".","F": "..－.",
              "G": "－－.","H": "....","I": "..","J": ".－－－","K": "－.－","L": ".－..",
              "M": "－－","N": "－.","O": "－－－","P": ".－－.","Q": "－－.－","R": ".－.",
              "S": "...","T": "－","U": "..－","V": "...－","W": ".－－","X": "－..－","Y": "－.－－","Z": "－－.."}
    
    f = open('dictionary.txt', 'r')
    words = [i.lower() for i in f.read().split()]
    f.close()

    # 文字の区切りを
    while True:


    ans = "pour"
    if ans in words:
        print(ans)


if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)
