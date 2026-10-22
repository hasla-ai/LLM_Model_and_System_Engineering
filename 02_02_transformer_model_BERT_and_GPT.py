# 02_02_BERT
# BERT builds contextual token representations with bidirectional self-attention over the full input sequence.
# This compact BERT-style model adds segment embeddings and provides masked-token and sentence-pair prediction heads.

# 입력 토큰 전체를 양방향 셀프 어텐션으로 처리해 문맥 표현을 만들고, 토큰·위치·세그먼트 임베딩을 더함.
# 프로젝트의 RMSNorm과 ReLU를 사용하고 패딩 마스크·드롭아웃·학습 루프는 포함하지 않는 간략화된 구조

class BERT:
    """A compact BERT-style encoder with masked-language-model and NSP output heads.

    This teaching implementation uses the project's RMSNorm and ReLU components,
    and does not include padding masks, dropout, or a training loop.
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_len):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        # --- 임베딩 (2.1절과 유사하지만 세그먼트 임베딩이 추가됨) ---

        self.wte = matrix(vocab_size, n_embd)   # 토큰 임베딩
        self.wpe = matrix(max_len, n_embd)      # 위치 임베딩
        self.wse = matrix(2, n_embd)            # ★ 세그먼트 임베딩 (문장 A=0, 문장 B=1)

        # --- 인코더 블록 × n_layer ---
        self.layers = []
        for _ in range(n_layer):
            layer = {
                'wq': matrix(n_embd, n_embd), 'wk': matrix(n_embd, n_embd),
                'wv': matrix(n_embd, n_embd), 'wo': matrix(n_embd, n_embd),
                'fc1': matrix(4 * n_embd, n_embd),
                'fc2': matrix(n_embd, 4 * n_embd),
            }
            self.layers.append(layer)

        # --- 출력 헤드 ---
        self.mlm_head = matrix(vocab_size, n_embd)  # [MASK] 위치 토큰 예측
        self.cls_head = matrix(2, n_embd)            # NSP 분류 (이어짐/안이어짐)

    def forward(self, token_ids, segment_ids):
        # Combine token identity, sequence position, and sentence segment information.
        xs = []
        for pos, (tok_id, seg_id) in enumerate(zip(token_ids, segment_ids)):
            tok_emb = self.wte[tok_id]
            pos_emb = self.wpe[pos]
            seg_emb = self.wse[seg_id]  # ★ 2.1절에는 없던 세그먼트 임베딩
            x = [t + p + s for t, p, s in zip(tok_emb, pos_emb, seg_emb)]
            xs.append(x)

        # Apply bidirectional encoder blocks; each token can attend to every input position.
        for layer in self.layers:
            new_xs = []
            for i, x in enumerate(xs):
                x_norm = rmsnorm(x)
                # ★ 셀프 어텐션: xs 전체를 참조 (양방향)
                attn_out = multi_head_attention(
                    x_norm, xs, xs,
                    layer['wq'], layer['wk'], layer['wv'], layer['wo']
                )
                x_res = [a + b for a, b in zip(attn_out, xs[i])]
                x_norm = rmsnorm(x_res)
                h = linear(x_norm, layer['fc1'])
                h = [hi.relu() for hi in h]
                h = linear(h, layer['fc2'])
                x_out = [a + b for a, b in zip(h, x_res)]
                new_xs.append(x_out)
            xs = new_xs
        return xs

    def predict_mask(self, xs, mask_positions):
        # Produce vocabulary scores for the selected masked-token positions.
        return [linear(xs[pos], self.mlm_head) for pos in mask_positions]

    def predict_nsp(self, xs):
        # Classify the sentence-pair relationship from the [CLS] representation.
        return linear(xs[0], self.cls_head)


# 02_02_GPT
# GPT uses causal self-attention so each position can use only the current and preceding tokens to predict what comes next.
# Unlike BERT, this compact decoder has no segment embeddings or cross-attention and returns a vocabulary score vector per position.

# 현재 위치까지의 토큰만 참조하는 인과적 셀프 어텐션과 위치별 어휘 점수 출력을 설명


class GPT:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_len):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        # --- 임베딩 (세그먼트 임베딩 없음 — BERT와 다름) ---
        self.wte = matrix(vocab_size, n_embd)   # 토큰 임베딩
        self.wpe = matrix(max_len, n_embd)      # 위치 임베딩
        # ★ BERT와 달리 세그먼트 임베딩(wse)이 없다

        # --- 디코더 블록 × n_layer ---
        self.layers = []
        for _ in range(n_layer):
            layer = {
                # ★ 셀프 어텐션만 있음 (크로스 어텐션 없음)

                'wq': matrix(n_embd, n_embd), 'wk': matrix(n_embd, n_embd),
                'wv': matrix(n_embd, n_embd), 'wo': matrix(n_embd, n_embd),
                'fc1': matrix(4 * n_embd, n_embd),
                'fc2': matrix(n_embd, 4 * n_embd),
            }
            self.layers.append(layer)

        # --- 출력 헤드: 다음 토큰 예측 (2.1절의 lm_head와 동일 역할) ---
        self.lm_head = matrix(vocab_size, n_embd)

    def forward(self, token_ids):
        seq_len = len(token_ids)

        # 1단계: 임베딩 (토큰 + 위치, 세그먼트 없음)
        xs = []
        for pos, tok_id in enumerate(token_ids):
            tok_emb = self.wte[tok_id]
            pos_emb = self.wpe[pos]
            x = [t + p for t, p in zip(tok_emb, pos_emb)]
            xs.append(x)

        # 2단계: 디코더 블록 반복
        for layer in self.layers:
            new_xs = []
            for i in range(seq_len):
                x = xs[i]
                x_norm = rmsnorm(x)
                # ★ 인과적 마스킹: xs[:i+1]로 현재 위치까지만 참조

                # BERT는 xs 전체를 참조했던 것과 대조적
                context = [xs[j] for j in range(i + 1)]
                attn_out = multi_head_attention(
                    x_norm, context, context,
                    layer['wq'], layer['wk'], layer['wv'], layer['wo']
                )
                x_res = [a + b for a, b in zip(attn_out, xs[i])]
                x_norm = rmsnorm(x_res)
                h = linear(x_norm, layer['fc1'])
                h = [hi.relu() for hi in h]
                h = linear(h, layer['fc2'])
                x_out = [a + b for a, b in zip(h, x_res)]
                new_xs.append(x_out)
            xs = new_xs
        return xs

    def predict_next(self, xs):
        # 각 위치에서 다음 토큰을 예측
        return [linear(x, self.lm_head) for x in xs]
