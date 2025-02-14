import streamlit as st
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from huggingface_hub import snapshot_download
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


def get_index_from_huggingface() :
    repo_id = 'codekookiz/manual-index'
    local_dir = "./manual_index_storage"

    token=get_huggingface_token()

    snapshot_download(
        repo_id=repo_id,        
        local_dir=local_dir,
        repo_type='dataset',
        token=token
    )

    # 다운로드한 폴더를 메모리에 로드
    storage_context = StorageContext.from_defaults(persist_dir= local_dir)

    index = load_index_from_storage(storage_context)


def main() :
    pass
    # 1. 사용할 모델 세팅
    # 2. 사용할 토크나이저 세팅 : embed_model
    initialize_models()

    # 3. RAG에 필요한 인덱스 세팅
    index = get_index_from_huggingface()

    # 4. 유저에게 프롬프트 입력 받아서 응답
    st.title('PDF 문서 기반 QnA 시스템')
    st.text('선진기업복지 업무매뉴얼을 기반으로 QnA를 제공합니다.')

    query_engine = index.as_query_engine()
    
    prompt = st.text_input('질문을 입력하세요.')

    if prompt :
        with st.spinner('답변을 생성하고 있습니다...') :
            response = query_engine.query(prompt)
            st.text('답변 : ')
            st.info(response.response)



if __name__ == '__main__' :
    main()