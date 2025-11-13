# 🤖 인공지능 PDF Q&A 챗봇
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda

# ==================== 1️⃣ 환경 설정 ====================
load_dotenv()

# LLM (OpenAI GPT-4o-mini)
llm = ChatOpenAI(model="gpt-4o-mini")

# 텍스트 분리기
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# 임베딩 모델
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}
)

# 프롬프트 템플릿
message = """
당신은 사용자의 질문에 답변을 하는 친절한 AI 어시스턴트입니다.
당신의 임무는 주어진 문맥을 토대로 사용자 질문에 답하는 것입니다.
만약 문맥에서 답변을 위한 정보를 찾을 수 없다면 
`주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다.` 라고 답하세요.
정보를 찾을 수 있다면 한글로 답변해 주세요.

## 주어진 문맥:
{context}

## 사용자 질문:
{input}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("human", message)
])

parser = StrOutputParser()

# 전역 변수
db = None
retriever = None
rag_chain = None


# ==================== 2️⃣ PDF 업로드 처리 함수 ====================
def load_pdf(file):
    global db, retriever, rag_chain

    print("📂 file:", file)
    if not file:
        return "⚠️ 파일을 업로드해주세요."

    try:
        # Gradio 5.x에서는 파일 경로가 문자열로 전달됨
        if isinstance(file, str):
            file_path = file
        elif hasattr(file, 'path'):
            file_path = file.path
        elif hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = str(file)

        print("📂 file_path:", file_path)

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"✅ PDF 페이지 수: {len(docs)}")

        docs = text_splitter.split_documents(docs)
        db = FAISS.from_documents(docs, hf_embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # Document 리스트를 문자열로 변환하는 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # input에서 질문을 추출하는 함수
        def get_question(input_dict):
            return input_dict["input"] if isinstance(input_dict, dict) else input_dict

        rag_chain = (
            RunnableMap({
                "context": RunnableLambda(get_question) | retriever | RunnableLambda(format_docs),
                "input": RunnablePassthrough(),
            })
            | prompt_template
            | llm
            | parser
        )

        print("✅ RAG chain 생성 완료")
        return "✅ PDF 파일이 성공적으로 업로드 및 처리되었습니다."

    except Exception as e:
        print("❌ PDF 처리 중 오류:", e)
        import traceback
        traceback.print_exc()
        return f"❌ 오류 발생: {str(e)}"


# ==================== 3️⃣ 질문 응답 처리 함수 ====================
def answer_question(question):
    if rag_chain is None:
        return "⚠️ PDF 파일을 먼저 업로드하세요."
    if not question:
        return "⚠️ 질문을 입력해주세요."
    try:
        return rag_chain.invoke({"input": question})
    except Exception as e:
        print("❌ 질문 처리 중 오류:", e)
        import traceback
        traceback.print_exc()
        return f"❌ 오류 발생: {str(e)}"


# ==================== 4️⃣ Gradio UI 구성 ====================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI PDF Q&A 챗봇
    **PDF 파일을 업로드하고 질문을 입력하면 AI가 답변을 제공합니다!**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="PDF 파일 업로드")
            upload_button = gr.Button("📥 업로드 및 처리")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="상태 메시지")
            question_input = gr.Textbox(label="질문 입력", placeholder="질문을 입력하세요.")
            submit_button = gr.Button("💬 답변 받기")
            answer_output = gr.Textbox(label="AI 답변")

    # 버튼 동작 연결
    upload_button.click(load_pdf, inputs=file_input, outputs=status_output)
    submit_button.click(answer_question, inputs=question_input, outputs=answer_output)

demo.launch(show_error=True)

 