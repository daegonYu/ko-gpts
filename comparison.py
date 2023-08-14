
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training


def load_model(model_id):
    
    print(model_id)

    bnb_config = BitsAndBytesConfig(    # 모델의 성능을 유지하면서 메모리 사용을 최적화하고, 하드웨어 환경에 맞게 데이터를 처리하는 데 도움
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')

    # inference 시에는 불필요
    # if model_id !=  'beomi/KoAlpaca-KoRWKV-6B':
    #     # 모델의 훈련 과정에서 필요한 메모리 양을 줄일 수 있습니다.
    #     model.gradient_checkpointing_enable()           # Activates gradient checkpointing for the current model.
    #     model = prepare_model_for_kbit_training(model)
        
    model.eval()
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
    
    return model, tokenizer


    
def gen(model, model_id, tokenizer, user_input, max_new_tokens=256):     # gen 의 형식은 모델마다 다를 수 있으니 참고하여 프롬프트를 설정해주세요.

    if model_id ==  'beomi/KoAlpaca-KoRWKV-6B':
        eos_token_id = 0
    else:
        eos_token_id = 2
    
    gened = model.generate(
        **tokenizer(
            f"### 질문: {user_input}\n\n### 답변:",
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        do_sample=True,
        eos_token_id=eos_token_id,
    )
    return tokenizer.decode(gened[0])

    
# def generate_response(model, tokenizer, input_text):
#     inputs = tokenizer.encode(input_text, return_tensors="pt")
#     outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response


KoRWKV, KoRWKV_tokenizer = load_model('beomi/KoAlpaca-KoRWKV-6B')
Llama_2_ko_7b, Llama_2_ko_7b_tokenizer = load_model("beomi/llama-2-ko-7b")
Llama_2_ko_7b_Chat, Llama_2_ko_7b_Chat_tokenizer = load_model('kfkas/Llama-2-ko-7b-Chat')
polyglot_ko, polyglot_ko_tokenizer = load_model("EleutherAI/polyglot-ko-5.8b")
polyglot_ko_kullm, polyglot_ko_kullm_tokenizer = load_model('nlpai-lab/kullm-polyglot-5.8b-v2')

KoRWKV_config = {
    'model' : KoRWKV,
    'tokenizer' : KoRWKV_tokenizer,
    'model_id' : 'beomi/KoAlpaca-KoRWKV-6B'
}

Llama_2_ko_7b_config = {
    'model' : Llama_2_ko_7b,
    'tokenizer' : Llama_2_ko_7b_tokenizer,
    'model_id' : "beomi/llama-2-ko-7b"
}

Llama_2_ko_7b_Chat_config = {
    'model' : Llama_2_ko_7b_Chat,
    'tokenizer' : Llama_2_ko_7b_Chat_tokenizer,
    'model_id' : 'kfkas/Llama-2-ko-7b-Chat'
}


polyglot_ko_config = {
    'model' : polyglot_ko,
    'tokenizer' : polyglot_ko_tokenizer,
    'model_id' : "EleutherAI/polyglot-ko-5.8b"
}


polyglot_ko_kullm_config = {
    'model' : polyglot_ko_kullm,
    'tokenizer' : polyglot_ko_kullm_tokenizer,
    'model_id' : 'nlpai-lab/kullm-polyglot-5.8b-v2'
}

model_ids = [KoRWKV_config['model_id'],Llama_2_ko_7b_config['model_id'],Llama_2_ko_7b_Chat_config['model_id'],polyglot_ko_config['model_id'],polyglot_ko_kullm_config['model_id']]

def main():
    st.title('Hello Everyone. I try comparison with 5 different models')
    
    responses = []
    
    # 사용자에게 입력받는 문장
    user_input = st.text_input("Enter your question:")
    
    if user_input:
        KoRWKV_answer = gen(model=KoRWKV_config['model'], model_id=KoRWKV_config['model_id'], tokenizer=KoRWKV_config['tokenizer'], user_input=user_input)
        Llama_2_ko_7b_answer = gen(model=Llama_2_ko_7b_config['model'], model_id=Llama_2_ko_7b_config['model_id'], tokenizer=Llama_2_ko_7b_config['tokenizer'], user_input=user_input)
        Llama_2_ko_7b_Chat_answer = gen(model=Llama_2_ko_7b_Chat_config['model'], model_id=Llama_2_ko_7b_Chat_config['model_id'], tokenizer=Llama_2_ko_7b_Chat_config['tokenizer'], user_input=user_input)
        polyglot_ko_answer = gen(model=polyglot_ko_config['model'], model_id=polyglot_ko_config['model_id'], tokenizer=polyglot_ko_config['tokenizer'], user_input=user_input)
        polyglot_ko_kullm_answer = gen(model=polyglot_ko_kullm_config['model'], model_id=polyglot_ko_kullm_config['model_id'], tokenizer=polyglot_ko_kullm_config['tokenizer'], user_input=user_input)
        
        responses.append(KoRWKV_answer)
        responses.append(Llama_2_ko_7b_answer)
        responses.append(Llama_2_ko_7b_Chat_answer)
        responses.append(polyglot_ko_answer)
        responses.append(polyglot_ko_kullm_answer)
        
        # 답변들을 출력
        for id, response in zip(responses,model_ids):
            st.write(f"{id} : response: {response}")

if __name__ == "__main__":
    main()