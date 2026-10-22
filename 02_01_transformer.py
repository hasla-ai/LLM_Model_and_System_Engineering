# 02_1_1

import math
import random
import sys
sys.setrecursionlimit(50000)
random.seed(42)

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 이 노드의 실제 숫자 값
        self.grad = 0                   # 이 노드에 대한 기울기 (역전파에서 계산됨)
        self._children = children       # 이 노드를 만든 입력 노드들
        self._local_grads = local_grads # 입력 노드 각각에 대한 국소 미분값

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self): return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))

    def exp(self):
        v = max(-20, min(20, self.data))
        return Value(math.exp(v), (self,), (math.exp(v),))

    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1

    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return other + (-self)

    def __rmul__(self, other): return self * other

    def __truediv__(self, other): return self * other**-1

    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

data_pairs = [
    ("i love you", "나는 너를 사랑해"),
    ("thank you", "고마워"),
    ("hello", "안녕"),
    ("good morning", "좋은 아침"),
    ("sorry", "미안해"),
    ("yes", "응"),
    ("no", "아니"),
    ("help me", "도와줘"),
    ("i am happy", "나는 행복해"),
    ("good night", "잘 자"),
    ("see you", "또 봐"),
    ("i am sad", "나는 슬퍼"),
    ("i am hungry", "나는 배고파"),
    ("water please", "물 주세요"),
    ("come here", "이리 와"),
    ("go away", "저리 가"),
    ("i know", "나는 알아"),
    ("i see", "알겠어"),
    ("be quiet", "조용히 해"),
    ("sit down", "앉아"),
]
print(f"데이터셋 크기: {len(data_pairs)}개") # 20개 
print(f"예시: '{data_pairs[0][0]}' → '{data_pairs[0][1]}'") # 예시: 'i love you' → '나는 너를 사랑해'

all_chars = sorted(set(''.join([src + tgt for src, tgt in data_pairs])))
PAD, BOS, EOS = 0, len(all_chars)+1, len(all_chars)+2
char_to_id = {ch: i+1 for i, ch in enumerate(all_chars)}
id_to_char = {i+1: ch for i, ch in enumerate(all_chars)}
id_to_char[PAD] = ''; id_to_char[BOS] = ''; id_to_char[EOS] = ''
vocab_size = len(all_chars) + 3

def tokenize(s):
    return [char_to_id[ch] for ch in s]

def detokenize(ids):
    return ''.join(id_to_char.get(i, '?') for i in ids if i not in (PAD, BOS, EOS))

max_enc_len = max(len(src) for src, _ in data_pairs) + 1
max_dec_len = max(len(tgt) for _, tgt in data_pairs) + 2

print(f"어휘 크기: {vocab_size} (문자 {len(all_chars)}개 + PAD + BOS + EOS)")
print(f"토큰화 예시: 'hello' → {tokenize('hello')}")
print(f"역토큰화:    {tokenize('hello')} → '{detokenize(tokenize('hello'))}'")
