이 레퍼지토리는 한국어 GPT 모델들을 직접 사용해보고 비교해 볼 수 있도록 한국어 GPT 모델들을 모아놓은 레퍼지토리입니다. 라이센스는 모두 apache-2.0 나 MIT로 상업적으로 사용가능합니다.

1. 각각의 모델들은 모델 이름으로 적힌 ipynb 파일로 따로 돌릴 수 있으며 모델들을 모아놓은 코드는 comparison.ipynb 파일입니다. (사용하실 때 GPU RAM 부족시 ipynb 상단의 Restart 해주세요.)
2. comparison.py는 streamlit으로 웹을 띄워서 각각의 모델을 비교할 수 있게 해놓았습니다.
3. finetune_kullm_polyglot.py는 그 중 성능이 제일 좋다고 생각한 nlpai-lab/kullm-polyglot-5.8b-v2 모델을 파인튜닝하는 코드입니다. 

<br>

### 참고한 GitHub, Hugging Face
1. https://github.com/nlpai-lab/KULLM
2. https://huggingface.co/EleutherAI/polyglot-ko-12.8b
3. https://huggingface.co/beomi/KoAlpaca-KoRWKV-6B
4. https://huggingface.co/beomi/llama-2-ko-7b
5. https://huggingface.co/kfkas/Llama-2-ko-7b-Chat
6. https://github.com/Beomi/KoAlpaca

<br>

### ipynb 별 설명
1. KoAlpaca-KoRWKV-6B.ipynb -> 모델 구조: RWKVv4 Neo Architecture(https://github.com/BlinkDL/RWKV-LM), 크기: 6B, 파인튜닝 데이터 : KoAlpaca Dataset(약 2만개 한국어 Q&A 데이터셋,https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)
2. KoAlpaca-Polyglot_6B.ipynb -> base : polyglot-ko-5.8b, 파인튜닝 데이터: KoAlpaca Dataset(약 2만개 한국어 Q&A 데이터셋)
3. Llama-2-ko-7b.ipynb -> 모델 구조 : llama-2, 크기: 7B, 파인튜닝 데이터: KoAlpaca Dataset(약 2만개 한국어 Q&A 데이터셋), 약 1.4만개 한국어 토큰 추가
4. Llama-2-ko-7b-Chat.ipynb -> base: beomi/llama-2-ko-7b, 파인튜닝 데이터: (약 15만개 한국어 Q&A 데이터(https://huggingface.co/datasets/nlpai-lab/kullm-v2)) 
5. polyglot-ko-5.8b.ipynb -> 모델 구조 : gpt-neo-x, 크기: 5.8B, 한국어 사전학습 모델로 사전학습 데이터 양: 167 billion tokens, 파인튜닝 : X
6. polyglot-KULLM_inference.ipynb -> finetune_kullm_polyglot.py를 통해 얻은 adapter_model.bin 파일을 이용하여 모델 추론해보는 코드
7. kullm.json -> finetune_kullm_polyglot.py으로 파인튜닝을 하면 prompter.py에 의해 kullm.json에 설정된 프롬프트 형식이 사용되는데 모델에 따라 정해진 프롬프트를 사용해야 최고 성능을 낼 수 있음(예를 들어 질 beomi의 모델들은 ### 질문: ### 답변: 의 형식이며 nlpai-lab의 모델은 ### 명령어: ### 응답: 의 프롬프트 형식을 따라야 함. 이것에 대해서는 comparison.ipynb 참고바랍니다. 모델마다 정해진 프롬프트를 사용하지 않으면 성능 차이가 많이 남.)
8. finetuninig.ipynb -> finetune_kullm_polyglot.py 은 프롬프트가 kullm 모델에 맞춰져 있는 반면 finetuninig.ipynb은 프롬프트가 그 외의 모델들에 맞춰져 있다.
9. ipynb 파일은 한국어로 파인튜닝 된 개별 모델을 실행시키는 코드이며 comparison.ipynb는 같은 질문에 대해 한국어 모델들의 성능을 서로 비교해보는 코드입니다.
10. comparison.py는 streamlit을 이용하여 웹에서 comparison.ipynb와 마찬가지로 한국어 모델의 성능을 비교해보는 코드입니다.<br>

### streamlit을 이용해서 웹으로 같은 질문에 대한 서로 다른 챗 모델 답변 비교해보기
(주의: 새로고침할때마다 GPU RAM을 중복해서 할당 받으므로 Ctrl+C로 종료하고 다시 실행하는 방식으로 해야함)

```python
streamlit run comparison.py
```

<br>

finetune_kullm_polyglot.py 를 이용하여 가장 괜찮은 성능을 뽐내는 kullm 모델을 자신의 데이터로 파인튜닝 할 수 있습니다.
- data_path -> 자신의 데이터 경로로 변경
- data 컬럼 : instruction, input, output
- instruction : 명령어, input : 예시, output : 답변 (참고 스탠퍼드에서 만든 Alpaca 데이터셋 의 경우 input이 들어간 데이터의 비율이 40%임. https://crfm.stanford.edu/2023/03/13/alpaca.html)
- data 형식 : jsonl
- 데이터 예시 : https://huggingface.co/datasets/nlpai-lab/kullm-v2
```python
finetune_kullm_polyglot.py \
--base_model='nlpai-lab/kullm-polyglot-5.8b-v2' \
--data_path='data.jsonl'    
```

<br>

finetuninig.ipynb에서는 질문과 답변 셋만 필요함(예시는 필요없음)

<br>

### 세부설명

polyglot-ko 모델 
- 한국어 데이터로 사전학습한 모델
- 모델 구조 : gpt-neo-x 모델 (https://github.com/EleutherAI/gpt-neox)
- 학습 데이터 : 863 GB의 한국어 데이터(주로 블로그, 뉴스) 셋 (167 billion tokens) (llama-2가 2 trillion tokens로 사전학습 했다는 것을 고려하면 1/10 이하의 데이터임.)
- 사전 학습 시 256개의 A100(80GB) 사용
- 성능 : 한국어 사전학습 모델 중 오픈된 모델 기준 가장 성능이 좋음(skt모델과 kakaobrain에서 만든 모델 보다 성능이 좋음)
- 라이센스 : apache-2.0 (상업적으로 사용가능) // 참고: kakaobrain 모델은 상업적으로 사용 불가

참고) https://huggingface.co/EleutherAI/polyglot-ko-12.8b

<br>

### 총평

1. 모델을 메모리 효율적으로 학습하기 위해 LoRA(LoRA: Low-Rank Adaptation of Large Language Models)의 방식으로 학습하다보니 사전학습 데이터의 중요성을 더더욱 크게 알게 되었다. 왜냐하면 LoRA로 학습하면 사전학습 모델의 파라미터는 고정되며 학습하다보니 모델의 답변이 사전학습의 데이터와 섞여서 나오는 현상을 발견할 수 있었으며 에폭을 늘리거나 데이터의 양을 늘려도 LoRA 학습의 특성상 사전학습의 데이터와 섞여나오는 것은 어쩔 수 없었다. 특정 포멧의 데이터로 학습하니 예를 들어 "A: ,B: " 이런 형식은 맞춰서 대답하는데 ":" 이후의 내용은 전부 틀린 답변을 내놓았다. 그렇다고 full finetuning을 하자니 계산상 적어도 100GB 이상의 GPU RAM이 필요했고 시간도 대단히 오래걸린다. 또한 EleutherAI/polyglot-ko 모델의 답변을 보면 학습 데이터의 질이 그렇게 좋지 않아보여 아쉬움이 남는다.

2. 영어로 사전학습된 모델(llama-2)의 한계 : 영어로 사전학습하다보니 한국어에 대한 이해가 부족하며 이를 극복하고자 한국어 token을 추가(약 1.4만개)하고 한국어 Q&A 데이터를 이용하여 LoRA로 파인튜닝해도 처음보는 질문에 대해서는 답변이 매우 이상했다. 하지만 학습한 질문에 대해서는 준수한 답변을 보였다. 한국어를 이해한다기 보다는 암기한 모델 같았다. 영어로 사전학습된 모델 + 한국어를 이해하기에는 적은 학습 데이터 개수 때문에 어쩔 수 없는 것 같다. 추가로 llama-1 모델을 62만개의 한국어 Q&A 데이터로 학습한 모델의 결과를 https://github.com/melodysdreamj/KoVicuna 에서 확인해 보길 바랍니다.

3. 챗봇 개발이 어려운 이유
   - 높은 진입 장벽 : 사전학습에 쓰이는 대단히 많은 데이터가 필요하고 무엇보다 많은 GPU가 필요함(llama-2의 경우 사전학습하는데 A100(80GB) 2048개가 쓰였으며 전기료만 수십억이 들었을 것으로 예상됨(https://moon-walker.medium.com/%EB%A6%AC%EB%B7%B0-meta-ai%EC%9D%98-small-gaint-model-llama-large-language-model-meta-ai-334e349ed06f)) 
    
4. 자신만의 특화된 데이터 필요 : 이루다를 만들기 위해서는 수많은 사용자의 대화 데이터가 필요하고 레시피 추천을 위해서는 많은 레시피 데이터가 필요하다. 따라서 갖고 있는 데이터가 곧 경쟁력이라 생각한다.

<br><br>

### https://github.com/nlpai-lab/KULLM 에서 아래 CODE를 가져왔습니다.

- finetune_kullm_polyglot.py
- merge_lora.py        

