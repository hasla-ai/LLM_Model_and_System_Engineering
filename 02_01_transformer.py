# 02_1_1
# A scalar value that records its inputs and local derivatives as a computation graph.
# Its backward pass propagates gradients through that graph using the chain rule.

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

# 02_1_2

# Create a small set of English-to-Korean sentence pairs for a toy translation task.
# These examples provide the parallel text data used by the model.

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
print(f"데이터셋 크기: {len(data_pairs)}개")
# 20개
print(f"예시: '{data_pairs[0][0]}' → '{data_pairs[0][1]}'")
# 예시: 'i love you' → '나는 너를 사랑해'

# 02_1_3
# Build a character-level vocabulary with PAD, BOS, and EOS special tokens.
# Add tokenization, detokenization, and sequence lengths for the translation data.

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
# 어휘 크기: 73 (문자 70개 + PAD + BOS + EOS)
print(f"토큰화 예시: 'hello' → {tokenize('hello')}")
# 토큰화 예시: 'hello' → [8, 6, 11, 11, 14]
print(f"역토큰화:    {tokenize('hello')} → '{detokenize(tokenize('hello'))}'")
# 역토큰화:    [8, 6, 11, 11, 14] → 'hello'

# 02_1_3
# Define linear, softmax, and RMSNorm utilities, then set the model dimensions.
# Initialize token and encoder/decoder position embedding matrices for the toy model.

# 핵심 함수 세 가지 (트랜스포머 전체에서 반복 사용)
def linear(x, w):
    """벡터 x에 가중치 행렬 w를 곱한다. 어텐션, MLP 등 모든 변환에 사용."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """숫자 목록을 확률 분포(합=1)로 변환한다. 어텐션 가중치와 출력 확률 계산에 사용."""
    max_val = max(v.data if isinstance(v, Value) else v for v in logits)
    exps = [(v - max_val).exp() if isinstance(v, Value) else Value(math.exp(v - max_val)) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """벡터의 크기를 정규화한다. 트랜스포머 블록 내부에서 값의 안정성을 유지하는 데 사용."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# 하이퍼파라미터
n_embd = 16                    # 임베딩 차원: 각 토큰을 표현하는 벡터의 크기
n_head = 4                     # 어텐션 헤드 수: 서로 다른 관점으로 문맥을 파악하는 병렬 어텐션
head_dim = n_embd // n_head    # 헤드당 차원: 16 ÷ 4 = 4

# 가중치 행렬 초기화 함수
def matrix(nout, nin, std=0.08):
    """nout×nin 크기의 가중치 행렬을 작은 난수로 초기화한다."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 임베딩 행렬
wte = matrix(vocab_size, n_embd)      # 토큰 임베딩: 각 토큰 ID → 16차원 벡터 (73×16)
wpe_enc = matrix(max_enc_len, n_embd) # 인코더 위치 임베딩: 각 위치 → 16차원 벡터
wpe_dec = matrix(max_dec_len, n_embd) # 디코더 위치 임베딩: 각 위치 → 16차원 벡터

# 02_1_4

# Look up and combine the token and position embeddings for the first character of "hello".
# Print sample values to verify the embedding lookup and vector addition.

tok_id = tokenize("h")[0]
tok_emb = wte[tok_id]
pos_emb = wpe_enc[0]
x = [t + p for t, p in zip(tok_emb, pos_emb)]

print(f"'h'의 토큰 ID: {tok_id}")
print(f"토큰 임베딩 (처음 4개): {[round(v.data, 4) for v in tok_emb[:4]]}")
print(f"위치 임베딩 (처음 4개): {[round(v.data, 4) for v in pos_emb[:4]]}")
print(f"합산 결과   (처음 4개): {[round(v.data, 4) for v in x[:4]]}")

# 'h'의 토큰 ID: 8
# 토큰 임베딩 (처음 4개): [0.1093, -0.1043, -0.0098, 0.0259]
# 위치 임베딩 (처음 4개): [0.0677, 0.0033, 0.0105, 0.0422]
# 합산 결과   (처음 4개): [0.1769, -0.101, 0.0007, 0.068]

# 02_1_5
# Initialize encoder, decoder, and output-head weight matrices for the transformer.
# Collect all embedding and model weights into one parameter list for training.

# 인코더 셀프 어텐션 가중치
enc_wq = matrix(n_embd, n_embd); enc_wk = matrix(n_embd, n_embd)
enc_wv = matrix(n_embd, n_embd); enc_wo = matrix(n_embd, n_embd)

# 인코더 MLP 가중치
enc_fc1 = matrix(4*n_embd, n_embd); enc_fc2 = matrix(n_embd, 4*n_embd)

# 디코더 셀프 어텐션 가중치
dec_wq = matrix(n_embd, n_embd); dec_wk = matrix(n_embd, n_embd)
dec_wv = matrix(n_embd, n_embd); dec_wo = matrix(n_embd, n_embd)

# 디코더 크로스 어텐션 가중치
cross_wq = matrix(n_embd, n_embd); cross_wk = matrix(n_embd, n_embd)
cross_wv = matrix(n_embd, n_embd); cross_wo = matrix(n_embd, n_embd)

# 디코더 MLP 가중치
dec_fc1 = matrix(4*n_embd, n_embd); dec_fc2 = matrix(n_embd, 4*n_embd)

# 출력 헤드 (디코더 출력을 어휘 크기의 확률 분포로 변환)
lm_head = matrix(vocab_size, n_embd)

# 모든 파라미터를 하나의 리스트로 모은다 (학습 시 일괄 업데이트를 위해)
all_matrices = [wte, wpe_enc, wpe_dec,
                enc_wq, enc_wk, enc_wv, enc_wo, enc_fc1, enc_fc2,
                dec_wq, dec_wk, dec_wv, dec_wo,
                cross_wq, cross_wk, cross_wv, cross_wo,
                dec_fc1, dec_fc2, lm_head]

params = [p for mat in all_matrices for row in mat for p in row]

print(f"전체 파라미터 수: {len(params)}") # 전체 파라미터 수: 9888

# 02_1_6

# Add encoder position embeddings to each character embedding in "hello".
# Project the first character's input vector into query, key, and value vectors.

# "hello"의 각 문자를 임베딩+위치인코딩

hello_tokens = tokenize("hello") # 문자 토큰 ID
hello_vecs = []
for pos, tid in enumerate(hello_tokens):
    te = wte[tid]   # 토큰 임베딩
    pe = wpe_enc[pos] # 인코더 위치 임베딩
    hello_vecs.append([t + p for t, p in zip(te, pe)]) # 입력 벡터 = 토큰 임베딩 + 인코더 위치 임베딩

# 'h'의 Q, K, V 생성

q0 = linear(hello_vecs[0], enc_wq)
k0 = linear(hello_vecs[0], enc_wk)
v0 = linear(hello_vecs[0], enc_wv)

print(f"'h'의 Q (처음 4개): {[round(v.data, 4) for v in q0[:4]]}")
print(f"'h'의 K (처음 4개): {[round(v.data, 4) for v in k0[:4]]}")
print(f"'h'의 V (처음 4개): {[round(v.data, 4) for v in v0[:4]]}")

# 'h'의 Q (처음 4개): [-0.0446, -0.0594, 0.0058, -0.0094]
# 'h'의 K (처음 4개): [-0.0135, 0.1147, -0.0469, -0.0175]
# 'h'의 V (처음 4개): [0.0111, 0.0075, 0.0316, 0.0137]

# 02_1_7
# Compute scaled dot-product attention scores for the first character against every character.
# Use the first attention head's query and keys; these logits are not softmax probabilities.

all_qs = [linear(xv, enc_wq) for xv in hello_vecs] # 각 문자입력 벡터에 각 가중치 적용한 Q, K, V 벡터.
all_ks = [linear(xv, enc_wk) for xv in hello_vecs]
all_vs = [linear(xv, enc_wv) for xv in hello_vecs]

q_h = all_qs[0][0:head_dim] # head_dim 설정값 4
k_h = [ki[0:head_dim] for ki in all_ks] # 각 문자의 K 벡터에서 헤드 0에 해당하는 앞의 4개 값

attn_logits = [ # 각 문자에 대한 attention logits 계산_ Q_h * K_h /
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
] # h가 h, e, l, l, o 각각에 주는 점수

print(f"'h'의 어텐션 점수 (헤드0, 스케일링 후):") # score.data는 자동미분용 Value 객체 안의 실제 숫자 값
for i, (ch, score) in enumerate(zip("hello", attn_logits)):
    print(f"  'h' → '{ch}': {round(score.data, 4)}")

# 'h'의 어텐션 점수 (헤드0, 스케일링 후):
#   'h' → 'h': -0.0032
#   'h' → 'e': 0.0017
#   'h' → 'l': -0.0004
#   'h' → 'l': 0.0008
#   'h' → 'o': -0.0014

# 02_1_08
# Apply softmax to the scaled attention logits to convert them into attention weights.
# The resulting weights form a probability distribution and sum to 1.

attn_weights = softmax(
    attn_logits
)
print(f"'h'의 어텐션 가중치 (헤드0, softmax 후):")
for ch, w in zip("hello", attn_weights):
    print(f"  'h' → '{ch}': {round(w.data, 4)}")
print(f"  합계: {round(sum(w.data for w in attn_weights), 4)}")
# 합계: 1.0

# 02_1_09

# Weight each character's value vector by its attention score and sum them.
# The result is the first character's context-aware output for attention head 0.

v_h = [vi[0:head_dim] for vi in all_vs]
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
print(f"'h'의 어텐션 출력 (헤드0): {[round(v.data, 4) for v in head_out]}")

# 'h'의 어텐션 출력 (헤드0): [-0.0084, -0.0168, 0.0014, 0.016]

# 02_1_10
# Compute scaled dot-product attention independently across all heads, then concatenate their outputs.
# Apply the output projection to return the combined attention result to the embedding dimension.

def multi_head_attention(q_in, k_list, v_list, wq, wk, wv, wo):
    """Multi-Head Attention.
    q_in: 쿼리 벡터 (16차원)
    k_list: 키 벡터들의 리스트 (각 16차원)
    v_list: 값 벡터들의 리스트 (각 16차원)
    wq, wk, wv: Q, K, V 가중치 행렬 (각 16×16)
    wo: 출력 가중치 행렬 (16×16)
    """
    q = linear(q_in, wq) # Q, K, V
    ks = [linear(ki, wk) for ki in k_list]
    vs = [linear(vi, wv) for vi in v_list]

    x_attn = []
    for h in range(n_head): # head가 4개.

        hs = h * head_dim
        q_h = q[hs:hs+head_dim]
        k_h = [ki[hs:hs+head_dim] for ki in ks]
        v_h = [vi[hs:hs+head_dim] for vi in vs] # 각각 attn_logits 계산.

        attn_logits = [
            sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
            for t in range(len(k_h))
        ]
        attn_weights = softmax(
            attn_logits
        )
        head_out = [
            sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) # 헤드 출력 이어 붙임.
            for j in range(head_dim)
        ]
        x_attn.extend(head_out)
    return linear(x_attn, wo) # 출력 가중치 적용.

# 02_1_11

# Run multi-head self-attention for each token in "hello", using the full sequence as keys and values.
# Collect and print one context-aware output vector per input token.

# hello"의 각 문자를 쿼리로 사용해 전체 문자열에 셀프 어텐션을 적용

print(f"입력: 'hello' ({len(hello_vecs)}개 토큰, 각 {len(hello_vecs[0])}차원)")

# 입력 토큰마다 하나씩, 총 5개의 16차원 벡터
mha_outputs = []
for i, xv in enumerate(hello_vecs):
    out = multi_head_attention(xv, hello_vecs, hello_vecs, enc_wq, enc_wk, enc_wv, enc_wo)
    mha_outputs.append(out)
    ch = "hello"[i]
    print(f"  '{ch}' MHA 출력 (처음 4개): {[round(v.data, 4) for v in out[:4]]}")
print(f"출력: {len(mha_outputs)}개 토큰, 각 {len(mha_outputs[0])}차원")

# 입력: 'hello' (5개 토큰, 각 16차원)
#   'h' MHA 출력 (처음 4개): [0.0002, 0.0015, 0.0014, 0.0039]
#   'e' MHA 출력 (처음 4개): [0.0002, 0.0015, 0.0013, 0.0038]
#   'l' MHA 출력 (처음 4개): [0.0002, 0.0015, 0.0014, 0.0038]
#   'l' MHA 출력 (처음 4개): [0.0002, 0.0015, 0.0014, 0.0038]
#   'o' MHA 출력 (처음 4개): [0.0002, 0.0015, 0.0013, 0.0038]

# 출력: 5개 토큰, 각 16차원

# 02_1_12

# RMSNorm 동작 확인 (일반 숫자로 원리 확인)

def rmsnorm_plain(x):
    """일반 float용 RMSNorm — 원리 확인용"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

x_big = [150.0, 280.0, 320.0]
x_big_after = rmsnorm_plain(x_big)
print(f"정규화 전: {x_big}")
print(f"정규화 후: {[round(v, 4) for v in x_big_after]}")

x_small = [0.01, 0.02, 0.03]
x_small_after = rmsnorm_plain(x_small)
print(f"정규화 전: {x_small}")
print(f"정규화 후: {[round(v, 4) for v in x_small_after]}")

# 정규화 전: [150.0, 280.0, 320.0]
# 정규화 후: [0.5762, 1.0756, 1.2293]
# 정규화 전: [0.01, 0.02, 0.03]
# 정규화 후: [0.458, 0.9161, 1.3741]

# 02_1_13
# A Transformer block combines attention for contextual understanding with an MLP for feature transformation.
# Residual connections preserve the input signal while normalization helps keep each sublayer stable.

# 트랜스포머 블록 동작 확인
# "hello"의 첫 문자 'h'에 대해 블록을 통과시킨다
x_input = hello_vecs[0]  # 'h'의 임베딩+위치 벡터 (16차원)

# --- 어텐션 블록 ---
x_residual = x_input                    # 잔차 연결용 보존
x = rmsnorm(x_input)                    # 정규화

# 멀티헤드 셀프 어텐션으로 주변 토큰의 문맥을 반영함.
attn_out = multi_head_attention(x, hello_vecs, hello_vecs, enc_wq, enc_wk, enc_wv, enc_wo)
x = [a + b for a, b in zip(attn_out, x_residual)]   # 잔차 연결: 어텐션 결과에 원래 입력 더함

# --- MLP 블록 ---
x_residual = x                          # 잔차 연결용 보존
x = rmsnorm(x)                          # 정규화
h = linear(x, enc_fc1)                  # 16차원 → 64차원
h = [hi.relu() for hi in h]             # ReLU
h = linear(h, enc_fc2)                  # 64차원 → 16차원
x_output = [a + b for a, b in zip(h, x_residual)]   # 잔차 연결

print(f"블록 입력 (처음 4개): {[round(v.data, 4) for v in x_input[:4]]}")
print(f"블록 출력 (처음 4개): {[round(v.data, 4) for v in x_output[:4]]}")
print(f"입력 차원: {len(x_input)}, 출력 차원: {len(x_output)}")

# 블록 입력 (처음 4개): [0.1769, -0.101, 0.0007, 0.068]
# 블록 출력 (처음 4개): [0.0878, 0.1029, -0.0786, 0.0801]
# 입력 차원: 16, 출력 차원: 16

# 02_1_14
# Convert token IDs into token-plus-position embeddings, then contextualize them with self-attention and an MLP.
# Residual connections preserve the input at each sublayer; this example uses one encoder block without padding masks.

# 입력 토큰 ID 목록을 받아, 각 토큰을 주변 토큰의 정보가 반영된 16차원 벡터로 바꿈.
# 현재 구현은 학습용 프레임워크의 완성된 인코더 스택이 아님
# 앞에서 만든 가중치를 이용해 인코더 블록 하나를 실행하는 작은 예제

# 패딩 마스크 구현하지 않음.

def encode(src_tokens):
    """인코더: 소스 토큰 시퀀스를 받아 문맥이 반영된 벡터 시퀀스를 반환한다.
    src_tokens: 토큰 ID 리스트 (예: [9, 1, 11, 14, 21, 6, 1, 24, 14, 20])
    반환: 각 토큰에 대한 16차원 벡터의 리스트
    """

    # 1단계: 각 토큰을 임베딩+위치 벡터로 변환

    xs = []
    for pos, tok_id in enumerate(src_tokens):
        tok_emb = wte[tok_id]        # 토큰 임베딩
        pos_emb = wpe_enc[pos]       # 위치 임베딩
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        xs.append(x)

    # 2단계: 인코더 블록 (셀프 어텐션 + MLP + 잔차 연결(residual connection))

    new_xs = []
    for i, x in enumerate(xs):
        x = rmsnorm(x)
        # 셀프 어텐션: 이 토큰의 Q로, 모든 토큰의 K, V를 참조
        attn_out = multi_head_attention(x, xs, xs, enc_wq, enc_wk, enc_wv, enc_wo)
        x_res = [a + b for a, b in zip(attn_out, xs[i])]  # 잔차 연결

        # MLP _ 특징 변환
        x_norm = rmsnorm(x_res)
        h = linear(x_norm, enc_fc1)       # 16 → 64: 차원 확장
        h = [hi.relu() for hi in h]       # ReLU (Value 객체의 메서드) : 음수 값을 0으로, 비선형 변환 추가.
        h = linear(h, enc_fc2)            # 64 → 16: 차원 확장 후 원복

        x_out = [a + b for a, b in zip(h, x_res)]  # 잔차 연결

        new_xs.append(x_out) # 토큰 별 결과 반환
    return new_xs

# 02_1_15
# An encoder transforms an input token sequence into contextual representations of its tokens.
# Each output vector keeps the embedding dimension while incorporating information from the sequence.

# "hello"를 인코더에 넣었을 때 입력 문자가 문맥이 반영된 벡터로 변환되는 결과를 보여줍니다.

# 인코더 실행 테스트
test_src = "hello"
src_tokens = tokenize(test_src)
enc_outputs = encode(src_tokens)

print(f"인코더 입력: '{test_src}' → 토큰 {src_tokens}")
print(f"인코더 출력: {len(enc_outputs)}개 벡터, 각 {len(enc_outputs[0])}차원")
for i, (ch, vec) in enumerate(zip(test_src, enc_outputs)):
    print(f"  '{ch}' 인코더 출력 (처음 4개): {[round(v.data, 4) for v in vec[:4]]}")

#   'h' 인코더 출력 (처음 4개): [0.0878, 0.1029, -0.0786, 0.0801]
#   'e' 인코더 출력 (처음 4개): [0.0652, -0.0891, 0.2288, -0.0413]
#   'l' 인코더 출력 (처음 4개): [-0.1629, -0.0291, -0.1405, -0.2295]
#   'l' 인코더 출력 (처음 4개): [-0.1973, 0.0547, -0.1306, -0.0783]
#   'o' 인코더 출력 (처음 4개): [0.2108, -0.0879, -0.3736, -0.0427]

# 02_1_16
# Decode one token at a time using causal self-attention over cached decoder states and cross-attention to encoder outputs.
# Pass the decoder state through an MLP and project it to vocabulary logits; logits are scores, not probabilities.

def decode_step(tgt_token_id, pos, dec_keys, dec_values, enc_outputs):

    """디코더가 토큰을 한 개씩 생성할 때 한 단계 처리하는 함수
    디코더 한 스텝: 현재 토큰 + 이전 디코더 상태 + 인코더 출력 → 다음 토큰 로짓

    tgt_token_id: 현재 입력 토큰 ID (BOS 또는 이전에 생성한 토큰)
    pos: 현재 위치 (0, 1, 2, ...)
    dec_keys: 이전 스텝들의 키 벡터 캐시 (리스트)
    dec_values: 이전 스텝들의 값 벡터 캐시 (리스트)
    enc_outputs: 인코더 출력 벡터 리스트
    반환: 73차원 로짓 (각 토큰이 다음에 올 확률 점수)
    """

    # 현재 토큰과 위치 정보를 임베딩하고 정규화

    tok_emb = wte[tgt_token_id]
    pos_emb = wpe_dec[pos]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # 1) 셀프 어텐션 (디코더 캐시에 현재까지 저장된 상태 즉 이전 디코더 토큰들만 참조 - 인과적 마스킹)
    dec_keys.append(x)
    dec_values.append(x)
    attn_out = multi_head_attention(x, dec_keys, dec_values, dec_wq, dec_wk, dec_wv, dec_wo)
    x = [a + b for a, b in zip(attn_out, x)]  # 잔차 연결

    # 2) 크로스 어텐션 (인코더 출력에서 입력 문맥 참조)
    x_res = x
    x = rmsnorm(x)
    cross_out = multi_head_attention(x, enc_outputs, enc_outputs, cross_wq, cross_wk, cross_wv, cross_wo)
    x = [a + b for a, b in zip(cross_out, x_res)]  # 잔차 연결

    # 3) MLP
    x_res = x
    x = rmsnorm(x)
    h = linear(x, dec_fc1)
    h = [hi.relu() for hi in h]
    h = linear(h, dec_fc2)
    x = [a + b for a, b in zip(h, x_res)]  # 잔차 연결

    # 4) 출력 헤드: 16차원 → 73차원 로짓
    logits = linear(x, lm_head)
    return logits

    # 검증_ 시작 토큰을 한 단계 처리했을 때 출력과 캐시의 크기가 예상과 같은지 확인

keys = []
values = []

logits = decode_step(BOS, 0, keys, values, enc_outputs)

print(len(logits))  # 73
print(len(keys), len(values))  # 1 1

# 02_1_17

# Train the encoder-decoder on sentence pairs with teacher forcing and next-token negative log-likelihood.
# Backpropagate each sequence loss and update all parameters with Adam while gradually lowering the learning rate.

# 번역 예제로 모델을 학습하는 루프.
# 매 단계에서 문장 쌍 하나를 골라 예측 오차를 계산하고, 그 오차가 줄어들도록 파라미터를 갱신.


# Adam 옵티마이저 버퍼 _ 학습률과 기울기의 이동 평균을 저장
learning_rate = 0.01
beta1, beta2, eps_adam = 0.9, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)

# 학습 루프
num_steps = 500
for step in range(num_steps):
    # 데이터 선택: 20개 쌍을 순환
    src, tgt = data_pairs[step % len(data_pairs)]
    src_tokens = tokenize(src)
    tgt_tokens = [BOS] + tokenize(tgt) + [EOS]  # 디코더 입력: BOS + 한국어 + EOS

    # 인코더 forward _ 영어 토큰을 문맥이 반영된 벡터로 변환
    enc_outputs = encode(src_tokens)

    # 디코더 forward: teacher forcing
    # 한국어 정답의 이전 토큰을 하나씩 디코더에 입력하여 다음 토큰을 예측
    # ex) BOS를 입력해 첫 글자를 예측하고, 이어 정답의 첫 글자를 넣어 다음 글자를 예측

    dec_keys, dec_values = [], []
    losses = []
    for pos in range(len(tgt_tokens) - 1):
        input_tok = tgt_tokens[pos]       # 현재 입력 (BOS, 나, 는, ...)
        target_tok = tgt_tokens[pos + 1]  # 예측해야 할 다음 토큰 (나, 는, 너, ...)
        logits = decode_step(input_tok, pos, dec_keys, dec_values, enc_outputs)
        probs = softmax(logits)
        loss_t = -probs[target_tok].log()  # 정답 토큰의 확률에 -log를 적용
        losses.append(loss_t)

    loss = sum(losses) / len(losses)  # 평균 손실

    # 역전파: 모든 파라미터의 기울기 계산
    for p in params:
        p.grad = 0       # 이전 스텝의 기울기 초기화
    loss.backward()      # 역전파 실행

    # Adam 옵티마이저: 기울기를 사용하여 파라미터 업데이트
    lr_t = learning_rate * (1 - step / num_steps)  # 학습률 선형 감소
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

# step    1/500 | loss 4.3864
# step   50/500 | loss 1.8143
# step  100/500 | loss 1.6139
# step  150/500 | loss 0.9630
# step  200/500 | loss 0.9197
# step  250/500 | loss 1.0082
# step  300/500 | loss 0.8272
# step  350/500 | loss 0.4460
# step  400/500 | loss 0.3542
# step  450/500 | loss 0.2803
# step  500/500 | loss 0.2035

# 02_1_18

# Run autoregressive inference on sample inputs, selecting the highest-scoring next token at each step.
# Stop at the end-of-sequence token or the decoder length limit, then detokenize and compare with the target.

# 앞의 10개 예문을 대상으로 번역을 생성함.
# BOS부터 시작해 매번 점수가 가장 높은 토큰을 고르는 greedy decoding을 수행하고,
# EOS가 나오거나 최대 길이에 도달하면 생성을 멈춰 정답과 비교.

print("\n--- 추론 결과 ---")
for src, tgt_expected in data_pairs[:10]:
    src_tokens = tokenize(src)
    enc_outputs = encode(src_tokens)

    dec_keys, dec_values = [], []
    generated = []
    tok_id = BOS   # 시작 토큰
    for pos in range(max_dec_len):
        logits = decode_step(tok_id, pos, dec_keys, dec_values, enc_outputs)
        # greedy decoding: 가장 높은 점수의 토큰을 선택
        tok_id = max(range(vocab_size), key=lambda i: logits[i].data)
        if tok_id == EOS:
            break
        generated.append(tok_id)

    result = detokenize(generated)
    print(f"  입력: '{src}' → 생성: '{result}' (정답: '{tgt_expected}')")


# --- 추론 결과 ---
#   입력: 'i love you' → 생성: '나는 너를 사랑해' (정답: '나는 너를 사랑해')
#   입력: 'thank you' → 생성: '나는 알아' (정답: '고마워')
#   입력: 'hello' → 생성: '안녕' (정답: '안녕')
#   입력: 'good morning' → 생성: '좋은 아침' (정답: '좋은 아침')
#   입력: 'sorry' → 생성: '미안해' (정답: '미안해')
#   입력: 'yes' → 생성: '응' (정답: '응')
#   입력: 'no' → 생성: '아니' (정답: '아니')
#   입력: 'help me' → 생성: '도와줘' (정답: '도와줘')
#   입력: 'i am happy' → 생성: '나는 슬퍼' (정답: '나는 행복해')
#   입력: 'good night' → 생성: '잘 자' (정답: '잘 자')

# ---------- 02_1 | Transformer: From Foundations to Training and Inference ----------