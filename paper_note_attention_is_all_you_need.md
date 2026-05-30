# Hiểu Transformer từ gốc: Attention, Multi-Head Attention, Embedding và Training Parallelism

Transformer là một trong những kiến trúc quan trọng nhất của deep learning hiện đại. Hầu hết LLM, VLM, ViT, Diffusion Transformer và VLA/robotics policy hiện nay đều kế thừa trực tiếp hoặc gián tiếp từ paper **Attention Is All You Need**.

Bài viết này không chỉ tóm tắt paper. Mục tiêu là giải thích Transformer theo hướng kỹ thuật, dễ triển khai:

- Tensor shape
- Toán học attention
- Embedding và positional encoding
- Encoder / decoder
- Training vs inference
- KV cache
- Optimization cho edge AI / robotics
- Code giả lập bằng PyTorch

## 1. Vấn đề Transformer giải quyết

Trước Transformer, các mô hình sequence-to-sequence mạnh thường dựa trên RNN, LSTM, GRU hoặc CNN.

Vấn đề lớn của RNN là xử lý sequence theo thứ tự:

```text
token 1 → token 2 → token 3 → ... → token n
```

Điều này tạo ra hai bottleneck lớn:

- **Khó parallel theo chiều sequence khi training** vì hidden state ở bước sau phụ thuộc hidden state bước trước.
- **Khó học long-range dependency** vì quan hệ xa giữa hai token phải truyền qua nhiều bước trung gian.

Transformer thay thế recurrence và convolution bằng **self-attention**.

Ý tưởng chính:

```text
Mỗi token có thể nhìn trực tiếp mọi token khác trong cùng sequence.
```

Trong paper, đây là thay đổi rất lớn: sequence modeling được chuyển từ pipeline tuần tự kiểu RNN sang các phép toán matrix multiplication lớn, rất hợp với GPU/TPU.

---

## 2. Tổng quan kiến trúc Transformer

### Chart — Luồng tổng thể Transformer encoder-decoder

![Chart — Luồng tổng thể Transformer encoder-decoder](https://mermaid.ink/img/Zmxvd2NoYXJ0IExSCiAgICBSYXdbIlJhdyB0ZXh0Il0gLS0-IFRva1siVG9rZW5pemVyIC8gQlBFIl0KICAgIFRvayAtLT4gSURzWyJUb2tlbiBJRHMiXQogICAgSURzIC0tPiBFbWJbIkVtYmVkZGluZyJdCiAgICBFbWIgLS0-IFBvc1siQWRkIHBvc2l0aW9uYWwgZW5jb2RpbmciXQogICAgUG9zIC0tPiBFbmNbIkVuY29kZXIgc3RhY2siXQogICAgRW5jIC0tPiBaWyJFbmNvZGVyIG91dHB1dCBaIl0KICAgIFByZWZpeFsiU2hpZnRlZCBvdXRwdXQgcHJlZml4Il0gLS0-IERlY0VtYlsiT3V0cHV0IGVtYmVkZGluZyBwbHVzIHBvc2l0aW9uIl0KICAgIERlY0VtYiAtLT4gRGVjWyJEZWNvZGVyIHN0YWNrIl0KICAgIFogLS0-IERlYwogICAgRGVjIC0tPiBMaW5lYXJbIkxpbmVhciBwcm9qZWN0aW9uIl0KICAgIExpbmVhciAtLT4gU29mdG1heFsiU29mdG1heCBvdmVyIHZvY2FidWxhcnkiXQogICAgU29mdG1heCAtLT4gTmV4dFsiTmV4dCB0b2tlbiBwcm9iYWJpbGl0aWVzIl0K?type=png)

Text được tokenize thành IDs, đi qua embedding + positional encoding, encoder tạo Z, decoder dùng shifted target prefix và Z để dự đoán next-token probabilities.

Transformer trong paper gốc vẫn là kiến trúc **encoder-decoder**.

```text
Input tokens
→ Input embedding
→ Add positional encoding
→ Encoder stack
→ Encoder output Z

Shifted output tokens
→ Output embedding
→ Add positional encoding
→ Decoder stack attends to Z
→ Linear
→ Softmax
→ Next-token probabilities
```

Encoder nhận input sentence và tạo representation `Z`.

Decoder nhận output prefix đã shift-right và dùng encoder output để dự đoán token tiếp theo.

Ví dụ dịch máy:

```text
Input:  I love robotics
Output: Tôi yêu robotics
```

Encoder xử lý:

```text
I love robotics → Z
```

Decoder sinh:

```text
<BOS> → Tôi → yêu → robotics → <EOS>
```

### Cấu hình trong paper

Transformer Base:

```text
N = 6 encoder layers + 6 decoder layers
d_model = 512
d_ff = 2048
num_heads = 8
d_k = d_v = 64
```

Transformer Big:

```text
N = 6
d_model = 1024
d_ff = 4096
num_heads = 16
```

---

## 3. Token, token ID và embedding

### Chart — Embedding lookup: token ID thành vector

![Chart — Embedding lookup: token ID thành vector](https://mermaid.ink/img/Zmxvd2NoYXJ0IFRECiAgICBUZXh0WyJSYXcgdGV4dDogSSBsb3ZlIHJvYm90aWNzIl0gLS0-IFRva2Vuc1siVG9rZW5zOiBJIHwgbG92ZSB8IHJvYm90aWNzIl0KICAgIFRva2VucyAtLT4gSURzWyJUb2tlbiBJRHM6IDEwIHwgMjUgfCA4OCJdCiAgICBJRHMgLS0-IE1hdHJpeFsiRW1iZWRkaW5nIG1hdHJpeCBFOiB2b2NhYl9zaXplIHggZF9tb2RlbCJdCiAgICBNYXRyaXggLS0-IFYxWyJFIHJvdyAxMDogdmVjdG9yIDUxMmQiXQogICAgTWF0cml4IC0tPiBWMlsiRSByb3cgMjU6IHZlY3RvciA1MTJkIl0KICAgIE1hdHJpeCAtLT4gVjNbIkUgcm93IDg4OiB2ZWN0b3IgNTEyZCJdCiAgICBWMSAtLT4gWFsiRW1iZWRkaW5nIG91dHB1dDogc2VxX2xlbiB4IGRfbW9kZWwiXQogICAgVjIgLS0-IFgKICAgIFYzIC0tPiBYCg?type=png)

Token ban đầu chỉ là index. Embedding matrix là bảng tham số trainable; lấy một dòng theo token ID để có vector d_model chiều.

Đây là chỗ rất dễ nhầm: text ban đầu không đi thẳng vào Transformer.

Pipeline đúng là:

```text
Text
→ Tokenizer / BPE / WordPiece
→ Token IDs
→ Embedding lookup
→ Token embedding vectors
→ Add positional encoding
→ Transformer
```

Ví dụ:

```text
"I love robotics"
```

Sau tokenizer/BPE:

```text
["I", "love", "robotics"]
```

Sau mapping sang ID:

```text
"I"        → 10
"love"     → 25
"robotics" → 88
```

Input IDs:

```python
input_ids = [10, 25, 88]
```

Token `"love"` ban đầu chỉ là index `25`.

Embedding layer là một bảng trainable:

```text
E ∈ R^{vocab_size × d_model}
```

Nếu:

```text
vocab_size = 37000
d_model = 512
```

thì:

```text
E.shape = [37000, 512]
```

Token `"love"` có ID 25:

```text
x_love = E[25]
```

Nghĩa là lấy dòng thứ 25 trong embedding matrix.

`E[25]` là vector 512 chiều:

```text
E[25] ∈ R^512
```

Các số trong vector này không do con người đặt. Chúng là tham số được học qua training.

### Code minh họa embedding lookup

```python
import torch
import torch.nn as nn

vocab_size = 37000
d_model = 512

embedding = nn.Embedding(vocab_size, d_model)

input_ids = torch.tensor([10, 25, 88])  # I love robotics

x = embedding(input_ids)

print(x.shape)
# torch.Size([3, 512])
```

Ở đây:

```text
input_ids.shape = [3]
embedding_output.shape = [3, 512]
```

Nếu có batch:

```text
input_ids.shape = [batch_size, seq_len]
embedding_output.shape = [batch_size, seq_len, d_model]
```

---

## 4. Positional Encoding: vì sao cần vị trí?

Self-attention tự bản thân không biết thứ tự token.

Ví dụ:

```text
robot picks cup
cup picks robot
```

Hai câu có cùng tập token nhưng nghĩa khác nhau.

Vì Transformer không có recurrence và convolution, nó phải được thêm thông tin vị trí.

Input vào encoder/decoder là:

```text
X_input = Embedding(tokens) * sqrt(d_model) + PositionalEncoding
```

Paper dùng sinusoidal positional encoding:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Trong đó:

```text
pos = vị trí token
i = index chiều embedding
d_model = 512
```

Có thể nhớ:

```text
Embedding cho biết token là gì.
Positional Encoding cho biết token đứng ở đâu.
```

Paper cũng thử learned positional embeddings và thấy kết quả gần như tương đương trên thí nghiệm của họ. Họ chọn sinusoidal vì hy vọng có thể extrapolate tốt hơn sang sequence dài hơn training.

---

## 5. Encoder layer: Self-Attention + FFN

Một encoder layer gồm 2 sub-layer:

```text
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network
```

Mỗi sub-layer có residual connection và LayerNorm.

Công thức trong paper:

```text
A = LayerNorm(X + MultiHeadAttention(X))
Y = LayerNorm(A + FFN(A))
```

Với Transformer Base:

```text
N = 6 layers
d_model = 512
d_ff = 2048
num_heads = 8
```

Trực giác:

- Self-attention trộn thông tin giữa token.
- FFN xử lý sâu representation của từng token sau khi đã nhận context.
- Residual connection giúp gradient đi qua mạng sâu ổn định hơn.
- LayerNorm giúp scale activation ổn định.

---

## 6. Decoder layer: Masked Self-Attention + Cross-Attention + FFN

Một decoder layer có 3 sub-layer:

```text
1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Attention / Cross-Attention
3. Position-wise Feed-Forward Network
```

Công thức:

```text
M = LayerNorm(Y + MaskedMHA(Y))
C = LayerNorm(M + CrossAttention(Q=M, K=Z, V=Z))
O = LayerNorm(C + FFN(C))
```

Trong đó:

```text
Z = encoder output
```

Decoder khác encoder ở chỗ có **causal mask** để không nhìn token tương lai.

Ví dụ training:

```text
Target:        Tôi yêu robotics <EOS>
Decoder input: <BOS> Tôi yêu robotics
Target label:  Tôi   yêu robotics <EOS>
```

Causal mask:

```text
              attend to
             <BOS>  Tôi  yêu  robotics
<BOS>          ✓     ✗    ✗       ✗
Tôi            ✓     ✓    ✗       ✗
yêu            ✓     ✓    ✓       ✗
robotics       ✓     ✓    ✓       ✓
```

Nếu không có causal mask, decoder trong training sẽ nhìn thấy đáp án tương lai, gây label leakage.

---

## 7. Attention: Query, Key, Value

Một attention function nhận:

```text
Query Q
Keys K
Values V
```

và tạo output.

Trực giác:

```text
Query = token hiện tại đang tìm gì
Key   = token khác có đặc điểm gì để được tìm thấy
Value = thông tin thật sự lấy về
```

Với một token `i`:

```text
q_i = x_i W_Q
k_i = x_i W_K
v_i = x_i W_V
```

Token `i` dùng query `q_i` để so với key của mọi token khác:

```text
score_ij = q_i · k_j
```

Sau softmax:

```text
α_ij = softmax_j(score_ij)
```

Output của token `i`:

```text
o_i = Σ_j α_ij v_j
```

Nghĩa là output là tổng có trọng số của các value vector.

---

## 8. Scaled Dot-Product Attention

### Chart — Scaled dot-product attention

![Chart — Scaled dot-product attention](https://mermaid.ink/img/Zmxvd2NoYXJ0IExSCiAgICBRWyJROiBxdWVyaWVzIl0gLS0-IFNjb3Jlc1siTWF0TXVsOiBRIHRpbWVzIEsgdHJhbnNwb3NlIl0KICAgIEtbIks6IGtleXMiXSAtLT4gU2NvcmVzCiAgICBTY29yZXMgLS0-IFNjYWxlWyJTY2FsZSBieSBzcXJ0IGRfayJdCiAgICBTY2FsZSAtLT4gTWFza1siQWRkIG9wdGlvbmFsIG1hc2siXQogICAgTWFzayAtLT4gU01bIlNvZnRtYXgiXQogICAgU00gLS0-IEFbIkF0dGVudGlvbiB3ZWlnaHRzIEEiXQogICAgQSAtLT4gT3V0WyJNYXRNdWw6IEEgdGltZXMgViJdCiAgICBWWyJWOiB2YWx1ZXMiXSAtLT4gT3V0CiAgICBPdXQgLS0-IE9bIkNvbnRleHQgb3V0cHV0Il0K?type=png)

Q so với K để tạo score, chia sqrt(d_k) để ổn định scale, optional mask chặn vị trí không hợp lệ, softmax tạo weight, rồi nhân với V để lấy weighted sum.

Công thức chính:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Shape

Giả sử sequence length là `n`, một head có dimension `d_k = d_v = 64`.

```text
Q ∈ R^{n × 64}
K ∈ R^{n × 64}
V ∈ R^{n × 64}
```

Tính score:

```text
QK^T ∈ R^{n × n}
```

Softmax tạo attention matrix:

```text
A = softmax(QK^T / sqrt(d_k))
A ∈ R^{n × n}
```

Nhân với value:

```text
O = A V
O ∈ R^{n × 64}
```

### Vì sao chia sqrt(d_k)?

Dot product:

```text
q · k = Σ_i q_i k_i
```

Nếu `d_k` lớn, tổng này có variance lớn hơn, score dễ trở nên rất lớn.

Khi score quá lớn, softmax trở nên quá sắc:

```text
softmax([20, 2, -5]) ≈ [1.0, 0.0, 0.0]
```

Gradient nhỏ, training khó ổn định.

Chia cho `sqrt(d_k)` giúp giữ scale ổn định:

```text
QK^T / sqrt(d_k)
```

Với paper base:

```text
d_k = 64
sqrt(d_k) = 8
```

Engineering interpretation: đây là một trick rẻ nhưng rất quan trọng để attention train ổn định ở hidden size thực tế.

---

## 9. Vì sao MatMul lại là Weighted Sum?

Paper viết:

```text
O = A V
```

Trong đó:

```text
A = softmax(QK^T / sqrt(d_k))
```

Nếu có 3 token:

```text
A =
[
  [α11, α12, α13],
  [α21, α22, α23],
  [α31, α32, α33]
]
```

Value matrix:

```text
V =
[
  v1,
  v2,
  v3
]
```

Khi nhân:

```text
O = A V
```

thì:

```text
o1 = α11v1 + α12v2 + α13v3
o2 = α21v1 + α22v2 + α23v3
o3 = α31v1 + α32v2 + α33v3
```

Vậy MatMul chính là cách viết gọn của nhiều phép weighted sum chạy song song.

---

## 10. Multi-Head Attention

### Chart — Multi-head attention shape flow

![Chart — Multi-head attention shape flow](https://mermaid.ink/img/Zmxvd2NoYXJ0IFRECiAgICBYWyJYOiBiYXRjaCB4IHNlcV9sZW4geCA1MTIiXSAtLT4gUUtWWyJMaW5lYXIgcHJvamVjdGlvbnM6IFdRIFdLIFdWIl0KICAgIFFLViAtLT4gU3BsaXRbIlNwbGl0IGhlYWRzOiBiYXRjaCB4IDggeCBzZXFfbGVuIHggNjQiXQogICAgU3BsaXQgLS0-IEgxWyJIZWFkIDEiXQogICAgU3BsaXQgLS0-IEgyWyJIZWFkIDIiXQogICAgU3BsaXQgLS0-IEgzWyIuLi4iXQogICAgU3BsaXQgLS0-IEg4WyJIZWFkIDgiXQogICAgSDEgLS0-IENhdFsiQ29uY2F0IGhlYWRzOiBiYXRjaCB4IHNlcV9sZW4geCA1MTIiXQogICAgSDIgLS0-IENhdAogICAgSDMgLS0-IENhdAogICAgSDggLS0-IENhdAogICAgQ2F0IC0tPiBXT1siT3V0cHV0IHByb2plY3Rpb24gV08iXQogICAgV08gLS0-IFlbIk1IQSBvdXRwdXQ6IGJhdGNoIHggc2VxX2xlbiB4IDUxMiJdCg?type=png)

Với Transformer Base: d_model=512, num_heads=8, head_dim=64. Các head chạy song song trong các subspace khác nhau rồi concat về 512 chiều.

Single-head attention chỉ có một “góc nhìn”.

Multi-head attention cho phép model học nhiều kiểu quan hệ khác nhau cùng lúc.

Công thức:

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

trong đó:

```text
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Với Transformer Base:

```text
h = 8
d_model = 512
d_k = d_v = 64
```

Mỗi head chiếu từ 512 chiều xuống 64 chiều:

```text
W_i^Q ∈ R^{512 × 64}
W_i^K ∈ R^{512 × 64}
W_i^V ∈ R^{512 × 64}
```

Output mỗi head:

```text
head_i ∈ R^{n × 64}
```

Concat 8 head:

```text
Concat(head_1, ..., head_8) ∈ R^{n × 512}
```

Sau đó qua output projection:

```text
W^O ∈ R^{512 × 512}
```

Kết quả cuối:

```text
MultiHead(Q, K, V) ∈ R^{n × 512}
```

### Vì sao multi-head tốt hơn single-head?

Một head đơn chỉ tạo một attention distribution. Nó giống như chỉ có một cách nhìn quan hệ giữa token.

Multi-head cho phép:

- Head này nhìn quan hệ gần.
- Head khác nhìn long-range dependency.
- Head khác học alignment source-target.
- Head khác học pattern cú pháp hoặc semantic.

Không có ai gán sẵn:

```text
Head 1 học syntax
Head 2 học object
Head 3 học long-range dependency
```

Các head học khác nhau vì:

```text
1. Mỗi head có W_Q, W_K, W_V riêng.
2. Khởi tạo random khác nhau.
3. Gradient update khác nhau.
4. W^O học cách khai thác từng head khác nhau.
```

Paper cũng có ablation: single-head attention kém hơn setting tốt nhất khoảng **0.9 BLEU**.

---

## 11. Ba cách Transformer dùng attention

Transformer dùng multi-head attention ở 3 nơi.

| Attention type | Q từ đâu | K từ đâu | V từ đâu | Mask? |
|---|---|---|---|---|
| Encoder self-attention | Encoder hidden | Encoder hidden | Encoder hidden | Không |
| Decoder masked self-attention | Decoder hidden | Decoder hidden | Decoder hidden | Có causal mask |
| Encoder-decoder attention | Decoder hidden | Encoder output | Encoder output | Không causal mask |

### Encoder self-attention

```text
Q = XW_Q
K = XW_K
V = XW_V
```

Mỗi input token nhìn mọi input token.

### Decoder masked self-attention

```text
Q = YW_Q
K = YW_K
V = YW_V
```

Nhưng có causal mask:

```text
O = softmax((QK^T / sqrt(d_k)) + mask) V
```

### Cross-attention

```text
Q = decoder_hidden W_Q
K = encoder_output W_K
V = encoder_output W_V
```

Decoder dùng query của mình để hỏi:

```text
Tôi đang sinh token này, cần nhìn phần nào của input?
```

---

## 12. Position-wise Feed-Forward Network

Sau attention, mỗi encoder/decoder layer có FFN.

Công thức:

```text
FFN(x) = max(0, xW1 + b1)W2 + b2
```

Với Transformer Base:

```text
d_model = 512
d_ff = 2048
```

Shape:

```text
x      ∈ R^512
W1     ∈ R^{512 × 2048}
hidden ∈ R^{2048}
W2     ∈ R^{2048 × 512}
output ∈ R^512
```

Tức là:

```text
512 → 2048 → 512
```

### Position-wise nghĩa là gì?

FFN xử lý từng token độc lập:

```text
f1 = FFN(h1)
f2 = FFN(h2)
f3 = FFN(h3)
```

Không có:

```text
f1 = FFN(h1, h2, h3)
```

Attention trộn thông tin giữa token. FFN xử lý sâu representation của từng token sau khi token đã nhận context.

---

## 13. Training Parallelism

### Chart — Training parallel vs inference autoregressive

![Chart — Training parallel vs inference autoregressive](https://mermaid.ink/img/Zmxvd2NoYXJ0IFRCCiAgICBzdWJncmFwaCBUcmFpblsiVHJhaW5pbmciXQogICAgICAgIFRhcmdldFsiRnVsbCB0YXJnZXQgc2VxdWVuY2UiXSAtLT4gU2hpZnRbIlNoaWZ0IHJpZ2h0Il0KICAgICAgICBTaGlmdCAtLT4gQ2F1c2FsWyJDYXVzYWwgbWFzayJdCiAgICAgICAgQ2F1c2FsIC0tPiBBbGxQb3NbIlByZWRpY3QgYWxsIHBvc2l0aW9ucyBpbiBvbmUgZm9yd2FyZCBwYXNzIl0KICAgIGVuZAogICAgc3ViZ3JhcGggSW5mZXJbIkluZmVyZW5jZSJdCiAgICAgICAgQk9TWyJCT1MiXSAtLT4gUzFbIlByZWRpY3QgdG9rZW4gMSJdCiAgICAgICAgUzEgLS0-IFMyWyJQcmVkaWN0IHRva2VuIDIiXQogICAgICAgIFMyIC0tPiBTM1siUHJlZGljdCB0b2tlbiAzIl0KICAgICAgICBTMyAtLT4gRU9TWyJDb250aW51ZSB1bnRpbCBFT1MiXQogICAgZW5kCiAgICBUcmFpbiAtLT4gTGVzc29uWyJUcmFpbmluZyBwYXJhbGxlbDsgaW5mZXJlbmNlIHNlcXVlbnRpYWwiXQogICAgSW5mZXIgLS0-IExlc3Nvbgo?type=png)

Training có target đầy đủ nên decoder tính nhiều vị trí cùng lúc bằng shifted-right + causal mask. Inference không có target thật nên vẫn phải sinh từng token.

Transformer parallelism trong training có nhiều chiều:

```text
1. Batch parallelism
2. Token parallelism
3. Head parallelism
```

### Batch parallelism

Nhiều sample được xử lý cùng lúc:

```text
X.shape = [batch_size, seq_len, d_model]
```

Ví dụ:

```text
X.shape = [2, 4, 512]
```

Nghĩa là:

```text
2 câu
4 token mỗi câu
512 chiều mỗi token
```

### Token parallelism

Trong self-attention, tất cả token trong sequence được xử lý cùng lúc bằng matrix multiplication:

```text
Q = XW_Q
K = XW_K
V = XW_V
scores = QK^T
```

Không cần loop tuần tự theo token như RNN:

```python
for token in sequence:
    ...
```

Paper so sánh số bước tuần tự tối thiểu:

```text
Self-attention: O(1)
RNN:            O(n)
```

### Head parallelism

Multi-head attention tính nhiều head cùng lúc.

Shape:

```text
Q.shape = [batch, num_heads, seq_len, head_dim]
```

Ví dụ:

```text
Q.shape = [2, 8, 4, 64]
```

Nghĩa là:

```text
2 samples
8 heads
4 tokens
64 dims/head
```

### Cái gì vẫn tuần tự?

Transformer không song song hoàn toàn ở mọi thứ.

```text
Layer 1 → Layer 2 → ... → Layer N
Encoder → Decoder
Decoder inference token 1 → token 2 → token 3
```

Training decoder song song được nhờ shifted-right target + causal mask.

Inference decoder vẫn phải sinh từng token một.

---

## 14. Training vs Inference

### Training

Training có target đầy đủ.

Ví dụ:

```text
Target:        Tôi yêu robotics <EOS>
Decoder input: <BOS> Tôi yêu robotics
Target label:  Tôi   yêu robotics <EOS>
```

Model tính probabilities cho toàn bộ sequence trong một forward pass:

```text
position 1 → predict Tôi
position 2 → predict yêu
position 3 → predict robotics
position 4 → predict <EOS>
```

### Inference

Inference không có target thật.

Model bắt đầu với:

```text
Prefix = [<BOS>]
```

Predict token tiếp theo:

```text
P(Tôi), P(Bạn), P(Anh), ..., P(token_V)
```

Model chọn token, ví dụ `"Tôi"`.

Prefix mới:

```text
[<BOS>, Tôi]
```

Tiếp tục predict:

```text
[<BOS>, Tôi] → yêu
[<BOS>, Tôi, yêu] → robotics
[<BOS>, Tôi, yêu, robotics] → <EOS>
```

Lưu ý: model không chỉ tính xác suất cho 3 từ minh họa. Nó tính xác suất cho toàn bộ vocabulary.

Nếu vocab có 37,000 token:

```text
logits ∈ R^{37000}
probs = softmax(logits)
```

Paper dùng beam search với beam size 4 và length penalty `alpha = 0.6` cho kết quả dịch máy.

---

## 15. KV Cache trong inference

Trong inference, nếu mỗi bước tính lại K/V cho toàn bộ prefix thì rất tốn.

Không cache:

```text
Step 3: tính lại K/V cho <BOS>, Tôi, yêu
```

Có KV cache:

```text
Step 3:
- dùng lại K/V của <BOS>, Tôi
- chỉ tính K/V mới cho yêu
```

Công thức:

```text
q_new = x_new W_Q
k_new = x_new W_K
v_new = x_new W_V
```

Append cache:

```text
K_cache = concat(K_cache, k_new)
V_cache = concat(V_cache, v_new)
```

Attention cho token mới:

```text
o_new = softmax(q_new K_cache^T / sqrt(d_k)) V_cache
```

Đây là lý do KV cache rất quan trọng trong LLM/VLA inference.

Lưu ý: paper gốc không tập trung vào KV cache như các LLM decoder-only hiện đại, nhưng bottleneck autoregressive decoding đã có sẵn trong kiến trúc decoder.

---

## 16. Regularization: Dropout và Label Smoothing

Paper dùng dropout và label smoothing để regularize model.

### Dropout

Dropout random tắt một phần activation trong training.

Với dropout rate:

```text
p_drop = 0.1
```

khoảng 10% activation bị set về 0.

Trong Transformer sub-layer:

```text
Without dropout:
LayerNorm(x + Sublayer(x))

With dropout:
LayerNorm(x + Dropout(Sublayer(x)))
```

Dropout cũng được apply lên:

```text
Embedding + Positional Encoding
```

Tức là:

```text
X = Dropout(Embedding(tokens) * sqrt(d_model) + PE)
```

Dropout giúp model không phụ thuộc quá mạnh vào một vài activation cụ thể.

### Label Smoothing

Bình thường target là one-hot.

Nếu target đúng là `"Tôi"`:

```text
Tôi = 1.0
các token khác = 0.0
```

Label smoothing làm mềm target:

```text
Tôi ≈ 0.9
các token khác chia nhau ≈ 0.1
```

Nó làm model bớt overconfident.

Paper ghi nhận label smoothing có thể làm perplexity xấu hơn, vì model học ít chắc chắn hơn, nhưng cải thiện accuracy và BLEU.

---

## 17. Performance chính của paper

Transformer Big đạt:

```text
WMT 2014 English-German: 28.4 BLEU
WMT 2014 English-French: 41.8 BLEU
```

Training setup quan trọng:

```text
Hardware: 8 NVIDIA P100 GPUs
Base model: ~12 giờ, 100K steps
Big model: ~3.5 ngày, 300K steps
```

Transformer không chỉ tốt hơn về chất lượng dịch, mà còn hiệu quả hơn về training cost so với nhiều baseline RNN/CNN trước đó.

Ý nghĩa engineering lớn nhất:

```text
Transformer biến sequence modeling thành matrix multiplication.
```

Đây là dạng computation rất hợp GPU/TPU.

---

## 18. Code: Scaled Dot-Product Attention

```python
import torch
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, heads, seq_q, head_dim]
    K: [batch, heads, seq_k, head_dim]
    V: [batch, heads, seq_k, head_dim]
    mask: broadcastable to [batch, heads, seq_q, seq_k]
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights
```

---

## 19. Code: Causal Mask

```python
import torch


def causal_mask(seq_len):
    """
    Returns lower-triangular mask.
    1 = allowed
    0 = blocked
    """
    return torch.tril(torch.ones(seq_len, seq_len))


mask = causal_mask(4)
print(mask)
```

Output:

```text
tensor([
 [1., 0., 0., 0.],
 [1., 1., 0., 0.],
 [1., 1., 1., 0.],
 [1., 1., 1., 1.]
])
```

---

## 20. Code: Multi-Head Attention Shape Flow

```python
import torch
import torch.nn as nn
import math


class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        # [batch, heads, seq_len, head_dim]
        return x

    def combine_heads(self, x):
        # x: [batch, heads, seq_len, head_dim]
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2)
        # [batch, seq_len, heads, head_dim]
        x = x.contiguous().view(batch_size, seq_len, heads * head_dim)
        # [batch, seq_len, d_model]
        return x

    def forward(self, X, mask=None):
        # X: [batch, seq_len, d_model]
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attn, V)
        concat = self.combine_heads(head_outputs)
        output = self.W_o(concat)

        return output
```

---

## 21. Code: Position-wise FFN

```python
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # output: [batch, seq_len, d_model]
        return self.net(x)
```

FFN không trộn token với nhau. Linear layer áp dụng trên chiều feature cuối cùng của từng token.

---

## 22. Code: Encoder Block

```python
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.mha = SimpleMultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

## 23. Code: Training vs Inference Prefix

```python
# Training
target = ["Tôi", "yêu", "robotics", "<EOS>"]

decoder_input = ["<BOS>", "Tôi", "yêu", "robotics"]
target_label  = ["Tôi", "yêu", "robotics", "<EOS>"]

# Model predicts all positions in parallel:
# <BOS>     -> Tôi
# Tôi       -> yêu
# yêu       -> robotics
# robotics  -> <EOS>


# Inference
prefix = ["<BOS>"]

# step 1
next_token = model_generate_next(prefix)
prefix.append(next_token)

# step 2
next_token = model_generate_next(prefix)
prefix.append(next_token)

# repeat until <EOS>
```

---

## 24. Vì sao Transformer quan trọng cho VLM/VLA/Robotics?

Trong robotics/VLA, input thường gồm nhiều loại token:

```text
language tokens
image tokens
robot state tokens
action tokens
```

Attention giúp model học quan hệ:

```text
instruction ↔ object in image
object ↔ action
robot state ↔ feasible motion
action history ↔ next action
```

Ví dụ:

```text
Instruction: Pick up the red cup
Image: camera observation
Robot state: joint positions
Output: action sequence
```

Multi-head attention có thể học nhiều quan hệ song song:

```text
Head 1: "red" ↔ visual region màu đỏ
Head 2: "cup" ↔ object token cái cốc
Head 3: "pick up" ↔ gripper action
Head 4: robot state ↔ action feasibility
```

Đây là lý do Transformer trở thành backbone cho rất nhiều VLM/VLA hiện đại.

---

## 25. Optimization cho edge AI

### Chart — Edge AI / robotics bottleneck map

![Chart — Edge AI / robotics bottleneck map](https://mermaid.ink/img/Zmxvd2NoYXJ0IExSCiAgICBUb2tlbnNbIkxhbmd1YWdlIGltYWdlIHN0YXRlIGFjdGlvbiB0b2tlbnMiXSAtLT4gU2VxWyJMb25nIHNlcXVlbmNlIGxlbmd0aCJdCiAgICBTZXEgLS0-IEF0dG5bIlF1YWRyYXRpYyBhdHRlbnRpb24gbWVtb3J5Il0KICAgIFNlcSAtLT4gS1ZbIktWIGNhY2hlIGJhbmR3aWR0aCJdCiAgICBEZWNvZGVbIkF1dG9yZWdyZXNzaXZlIGRlY29kaW5nIl0gLS0-IExhdGVuY3lbIkxhdGVuY3kgYm90dGxlbmVjayJdCiAgICBFZGdlWyJKZXRzb24gUXVhbGNvbW0gRWRnZSBHUFUiXSAtLT4gQ29uc3RyYWludHNbIlBvd2VyIGFuZCBtZW1vcnkgY29uc3RyYWludHMiXQogICAgQXR0biAtLT4gT3B0WyJPcHRpbWl6YXRpb24iXQogICAgS1YgLS0-IE9wdAogICAgTGF0ZW5jeSAtLT4gT3B0CiAgICBDb25zdHJhaW50cyAtLT4gT3B0CiAgICBPcHQgLS0-IFF1YW50WyJRdWFudGl6YXRpb24iXQogICAgT3B0IC0tPiBGbGFzaFsiRmxhc2hBdHRlbnRpb24iXQogICAgT3B0IC0tPiBTdGF0aWNbIlN0YXRpYyBzaGFwZSBPTk5YIFRlbnNvclJUIl0KICAgIE9wdCAtLT4gRnVzZWRbIkZ1c2VkIFFLViBMTiBNTFAiXQo?type=png)

Khi đưa Transformer lên robot/edge device, bottleneck thường là attention memory, autoregressive latency, KV cache bandwidth và dynamic shape.

Khi triển khai Transformer trên Jetson, Qualcomm, mobile GPU hoặc robot edge device, bottleneck thường không chỉ nằm ở số parameter.

Các bottleneck chính:

- Attention memory `O(n^2)` theo sequence length.
- Autoregressive decoding sinh từng token.
- KV cache tốn memory bandwidth.
- Softmax, LayerNorm và small-batch inference khó tận dụng accelerator tối đa.
- Vision tokens làm sequence length tăng rất nhanh.

Các hướng tối ưu:

- FP16 / BF16 trước, sau đó INT8 hoặc INT4 nếu accuracy cho phép.
- KV cache layout tối ưu cho decoding.
- FlashAttention hoặc memory-efficient attention.
- Fused QKV projection.
- Fused LayerNorm.
- Fused MLP.
- Static-shape ONNX / TensorRT export.
- Local/window attention nếu context quá dài.
- GQA/MQA để giảm KV cache trong decoder-only LLM hiện đại.

---

## 26. Những điểm cần nhớ

```text
Token ban đầu là index.
Embedding lookup biến index thành vector.
Positional encoding thêm thông tin vị trí.
Self-attention cho token nhìn token khác.
Scaled dot-product attention dùng QK^T / sqrt(d_k).
Softmax tạo attention weights.
Nhân với V tạo weighted sum.
Multi-head attention tạo nhiều góc nhìn song song.
FFN xử lý từng token độc lập sau khi token đã nhận context.
Training parallel theo batch, token, head.
Inference decoder vẫn sinh từng token.
KV cache rất quan trọng cho inference hiện đại.
Dropout và label smoothing giúp regularization.
```

---

## 27. Kết luận

Transformer mạnh không chỉ vì attention, mà vì toàn bộ thiết kế rất hợp với GPU:

```text
matrix multiplication
parallel token processing
multi-head computation
residual connections
layer normalization
position-wise FFN
```

Paper **Attention Is All You Need** đã biến sequence modeling từ pipeline tuần tự kiểu RNN thành kiến trúc có thể train song song mạnh mẽ.

Đây là nền tảng để hiểu các mô hình hiện đại như GPT, BERT, T5, ViT, CLIP, RT-1, RT-2, OpenVLA, GR00T và SmolVLA.

Nếu muốn biến paper này thành mini-project, hướng tốt nhất là:

```text
Minimal Transformer from scratch
→ copy task / toy translation
→ attention visualization
→ causal mask tests
→ KV cache
→ ONNX export
→ latency + memory benchmark
→ optimization report
```
