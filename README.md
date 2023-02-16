# learnGPT - team 1

## week 1

### BPE tokenization?

- how does BPE work?
- explain how it works by accounting for the A-shaped graph
- how is this better than character-level tokenization?

### `test_1.py`?

>  `GPTVer1` is a naive Bigram LM that performs poorly - why?

- "The quick brown fox jumps over the lazy:\nHAGdirdo sick's q-Whe,\n\nANs " - why gibberish at the end?
- In what ways bigarm LM's are limited by?

## week 2 - Introduction
week2에서는 week1에서 완성한 GPTVer3(one-head self-attention + positional encoding)와 HeadVer4(self-attention head)를 더 발전시켜 최종적으로 NanoGPT를 구현하는 것을 목표로 합니다.

## week 2 -  `test_7.py` (Team 1)

```shell
pytest tests/test_7.py -s -vv
```

### test_head_ver_4_and_multi_head_ver_1_are_equally_expensive & test_multi_head_helps

<img src='img/Multi-Head Attention.png' width=250>

Week1에서 구현했던 `HeadVer4`(self-attention head)를 바탕으로 multi-head attention을 구현합니다.
self-attention head에서 Q, K, V가 각각 FC layer를 통과하고나면 (batch_size, block_size, embed_size) → (batch_size, block_size, head_size)로 shape이 변경이 됩니다.
그리고 embed_size = head_size * n_heads의 관계가 성립합니다.

> TODO 1: `MultiHeaVer1.forward`를 구현해주세요.
> input x를 n_heads 개의 self-attention head를 통과한 후 head_output을 concatnate합니다. 그리고 projection layer(FC)를 통과시켜 multi-head attention을 구현해주세요.

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer1`와 `HeadVer4`의 연산량에 차이가 있나요? 없다면 왜?
2. `MultiHeadVer1`와 `HeadVer4` 중 어떤 것이 더 좋은 성능을 보이나요? 그 이유는 무엇인가요?

### test_multi_head_ver_2_is_faster_than_ver_1 & test_multi_head_ver_1_and_multi_head_ver_2_are_logically_equal

테스트를 돌려보고 다음의 질문에 답해주세요.
1. `MultiHeadVer2`와 `MultiHeadVer1`는 알고리즘에 차이는 없지만 `MultiHeadVer2`가 연산 속도가 더 빠릅니다. 그 이유는 무엇인가요?




