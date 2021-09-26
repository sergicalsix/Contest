import sys
import random 

## 空白のないモールス信号を翻訳する問題
def main(lines):
    s:str = lines[0]
   
    morse_dict_alpha = {"A": ".-","B": "-...","C": "-.-.","D": "-..","E": ".","F": "..-.",
              "G": "--.","H": "....","I": "..","J": ".---","K": "-.-","L": ".-..",
              "M": "--","N": "-.","O": "---","P": ".--.","Q": "--.-","R": ".-.",
              "S": "...","T": "-","U": "..-","V": "...-","W": ".--","X": "-..-","Y": "-.--","Z": "--.."}
    morse_code = {v:k.lower() for k,v in morse_dict_alpha.items()}

    f = open('dictionary.txt', 'r')
    words = [i.lower() for i in f.read().split()]
    f.close()

    # 文字の区切りをランダムにして翻訳していく
    
    while True:
        tmp_s = s
        encoded = ""
      
        ##文字の区切りを作成
        while len(tmp_s) > 0:
            max_len = len(tmp_s)
            if len(tmp_s) > 4: #モールス信号の長さの最大は4
                max_len = 4 
            a = random.randint(1, max_len)
            
            if tmp_s[:a] in morse_code.keys():
                encoded += morse_code[tmp_s[:a]]
                tmp_s = tmp_s[a:]
           
        if encoded in words:
            print(encoded)
            break

if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)
