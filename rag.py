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

# 프롬프트 템플릿 (대화 기록 포함)
system_message = """당신은 사용자의 질문에 답변을 하는 친절한 AI 어시스턴트입니다.
당신의 임무는 주어진 문맥을 토대로 사용자 질문에 답하는 것입니다.
만약 문맥에서 답변을 위한 정보를 찾을 수 없다면 
`주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다.` 라고 답하세요.
정보를 찾을 수 있다면 한글로 답변해 주세요.
이전 대화 내용을 참고하여 일관성 있는 답변을 제공하세요.

## 이전 대화:
{chat_history}

## 주어진 문맥:
{context}

## 사용자 질문:
{input}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("human", system_message)
])

parser = StrOutputParser()

# 전역 변수
db = None
retriever = None
rag_chain = None
chat_history = []  # 대화 기록 저장


# ==================== 2️⃣ PDF 업로드 처리 함수 ====================
def load_pdf(file, chat_history):
    global db, retriever, rag_chain

    print("📂 file:", file)
    if not file:
        return chat_history, "⚠️ 파일을 업로드해주세요."

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
                "chat_history": RunnablePassthrough(),
            })
            | prompt_template
            | llm
            | parser
        )

        print("✅ RAG chain 생성 완료")
        status_msg = "✅ PDF 파일이 성공적으로 업로드 및 처리되었습니다. 이제 질문을 입력하세요!"
        # 채팅에 시스템 메시지 추가
        chat_history.append(("", status_msg))
        return chat_history, status_msg

    except Exception as e:
        print("❌ PDF 처리 중 오류:", e)
        import traceback
        traceback.print_exc()
        error_msg = f"❌ 오류 발생: {str(e)}"
        return chat_history, error_msg


# ==================== 3️⃣ 질문 응답 처리 함수 (채팅 인터페이스) ====================
def add_message(message, history):
    """사용자 메시지를 채팅창에 즉시 추가"""
    if not message or not message.strip():
        return history, ""
    history.append((message, "💭 답변 생성 중..."))
    return history, ""


def chat_with_pdf(history):
    """PDF 기반으로 답변 생성"""
    global rag_chain
    
    if not history:
        return history
    
    # 마지막 메시지(사용자 질문)를 가져옴
    current_question = history[-1][0]
    
    if rag_chain is None:
        # 마지막 메시지의 답변 부분만 업데이트
        history[-1] = (current_question, "⚠️ PDF 파일을 먼저 업로드하세요.")
        return history
    
    if not current_question or not current_question.strip():
        return history
    
    try:
        # 대화 기록을 문자열로 변환 (마지막 메시지 제외)
        chat_history_str = ""
        if len(history) > 1:
            for human_msg, ai_msg in history[:-1]:
                if human_msg:
                    chat_history_str += f"사용자: {human_msg}\n"
                if ai_msg and not ai_msg.startswith("💭"):
                    chat_history_str += f"AI: {ai_msg}\n"
        
        # 현재 질문에 대한 답변 생성
        response = rag_chain.invoke({
            "input": current_question,
            "chat_history": chat_history_str if chat_history_str else "이전 대화가 없습니다."
        })
        
        # 마지막 메시지의 답변 부분만 업데이트
        history[-1] = (current_question, response)
        return history
    
    except Exception as e:
        print("❌ 질문 처리 중 오류:", e)
        import traceback
        traceback.print_exc()
        error_msg = f"❌ 오류 발생: {str(e)}"
        history[-1] = (current_question, error_msg)
        return history


# ==================== 4️⃣ 대화 기록 초기화 함수 ====================
def clear_chat():
    global chat_history
    chat_history = []
    return []


# ==================== 5️⃣ Gradio UI 구성 (채팅 인터페이스) ====================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI PDF Q&A 챗봇
    **PDF 파일을 업로드하고 채팅으로 질문하면 AI가 답변을 제공합니다!**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 PDF 파일 업로드", file_types=[".pdf"])
            upload_button = gr.Button("📥 업로드 및 처리", variant="primary")
            status_output = gr.Textbox(label="상태", interactive=False, lines=2)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 채팅",
                height=500,
                show_copy_button=True,
                avatar_images=(None, "🤖")
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="질문 입력",
                    placeholder="PDF에 대해 질문하세요...",
                    show_label=False,
                    scale=7
                )
                submit_button = gr.Button("전송", variant="primary", scale=1)
            clear_button = gr.Button("🗑️ 대화 기록 지우기", variant="secondary", size="sm")

    # 이벤트 연결
    upload_button.click(
        load_pdf,
        inputs=[file_input, chatbot],
        outputs=[chatbot, status_output]
    )
    
    # 질문 제출: 1) 질문 즉시 추가 -> 2) 답변 생성
    msg.submit(
        add_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    ).then(
        chat_with_pdf,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    submit_button.click(
        add_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    ).then(
        chat_with_pdf,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    clear_button.click(
        clear_chat,
        outputs=[chatbot]
    )

demo.launch(show_error=True)

 