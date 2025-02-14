import streamlit as st
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os


def get_huggingface_token() :
    token = st.secrets.get('HUGGINGFACE_API_TOKEN')
    return token


@st.cache_resource
def initialize_models() :
    # 허깅페이스에서 받아서 사용
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    token = get_huggingface_token()

    llm = HuggingFaceInferenceAPI(
        model_name=model_name,
        max_new_tokens=512,
        temperature=0,
        system_prompt="""
        당신은 한국어로 대답하는 AI 어시스턴트입니다.
        주어진 질문에 대해서만 한국어로 명확하고 정확하게 답변해주세요.
        응답의 마지막 부분은 단어로 끝내지 말고 문장으로 끝내도록 해주세요.
        """,
        token=token
    )

    embed_model_name = 'sentence-transformers/all-mpnet-base-v2'
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    Settings.llm = llm
    Settings.embed_model = embed_model


def main() :
    pass
    # 1. 사용할 모델 세팅
    # 2. 사용할 토크나이저 세팅 : embed_model
    # 3. RAG에 필요한 인덱스 세팅
    # 4. 유저에게 프롬프트 입력 받아서 응답
    initialize_models()

    st.title('PDF 문서 기반 QnA 시스템')
    st.text('선진기업복지 업무매뉴얼을 기반으로 QnA를 제공합니다.')


if __name__ == '__main__' :
    main()