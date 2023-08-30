import glob
import os
import re
from typing import Any, List, Set, Tuple
from urllib.parse import urljoin
import datetime

import dotenv
import openai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from trafilatura import html2txt


class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """日本語の文章を分割するクラス"""

    def __init__(self, **kwargs: Any) -> None:
        """初期化関数"""
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)


def init() -> None:
    """環境変数を初期化する関数"""
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


from typing import cast


def load_qdrant() -> Qdrant:
    """Qdrantをロードする関数。

    Returns:
        Qdrant: Qdrantのインスタンス
    """
    qdrant_path: str = cast(str, os.getenv("QDRANT_PATH"))
    collection_name: str = cast(str, os.getenv("COLLECTION_NAME"))
    client: QdrantClient = QdrantClient(path=qdrant_path)
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print("collection created")
    return Qdrant(client=client, collection_name=collection_name, embeddings=OpenAIEmbeddings())


def init_page() -> None:
    """Streamlitページの初期設定を行う関数"""
    st.set_page_config(page_title="ChatGPT", page_icon="🤗")
    st.header("ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages() -> None:
    """メッセージの初期設定を行う関数"""
    init_content = f"""
    You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=init_content)]
        st.session_state.costs = []


def select_model() -> ChatOpenAI:
    """モデルを選択する関数

    Returns:
        ChatOpenAI: 選択されたモデル
    """
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def show_massages(messages_container: Any) -> None:
    """メッセージを表示する関数

    Args:
        messages_container (st.container): Streamlitのコンテナ
    """
    messages = st.session_state.get("messages", [])
    with messages_container:
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")


def build_qa_model(llm: ChatOpenAI) -> Any:
    """質問応答モデルを構築する関数

    Args:
        llm (OpenAILLM): 言語モデル

    Returns:
        RetrievalQA: 質問応答モデル
    """
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever()
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in Japanese:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(  # noqa: C0103
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa


def query_search(query: str) -> List[Tuple[Document, float]]:
    """Qdrantを用いて、類似する文書を検索する関数

    Args:
        query (str): 検索クエリ

    Returns:
        List[Document]: 検索結果の文書リスト
    """
    qdrant = load_qdrant()
    docs = qdrant.similarity_search_with_score(query, k=2)
    del qdrant
    return docs


def place_input_form(
    input_container: Any,
    #  messages_container: Any,
    llm: ChatOpenAI,
) -> None:
    """入力フォームを配置する関数

    Args:
        input_container (st.container): 入力用のStreamlitコンテナ
        messages_container (st.container): メッセージ表示用のStreamlitコンテナ
        llm (OpenAILLM): 言語モデル
    """
    with input_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(label="Message: ", key="input")
            submit_button = st.form_submit_button(label="Send")
        if submit_button and user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                response = build_qa_model(llm).run(user_input)
            st.session_state.messages.append(AIMessage(content=response))


def build_vector_store() -> None:
    """ベクトルストアを構築する関数"""
    qdrant = load_qdrant()
    html_files: List[str] = glob.glob("documents/html/**/*.html", recursive=True)
    docs = []
    progress_bar = st.progress(0, text="Start")
    count = 0
    for html_file in html_files:
        # tmp_text.txtにhtmlファイルをテキスト化して保存する
        with open(html_file, "r", encoding="utf-8") as file:
            html = file.read()
        text = html2txt(html)
        with open("tmp_text.txt", "w", encoding="utf-8") as file:
            file.write(text)
        # テキストファイルを読み込む
        loader = TextLoader("tmp_text.txt")
        text_splitter = JapaneseCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=0,
        )
        split_documents = text_splitter.split_documents(loader.load_and_split())
        split_documents = [
            Document(
                page_content=modify_duplicate_sentence(split_document.page_content),
                metadata={"source": split_document.metadata["source"]},
            )
            for split_document in split_documents
        ]
        docs.extend(split_documents)
        os.remove("tmp_text.txt")
        progress_bar.progress(count / len(html_files), text=f"実行状況: {count}/{len(html_files)}")
        count += 1
    qdrant.add_documents(docs)
    del qdrant


def modify_duplicate_sentence(sentence: str) -> str:
    """重複する文字列を修正する関数

    Args:
        sentence (str): 入力文字列

    Returns:
        str: 修正された文字列
    """
    return re.sub(r"(.+?)\1+", r"\1", sentence)


def document_to_vector() -> None:
    """ドキュメントをベクトル化する関数"""
    st.write("docment配下のファイルをベクトル化します")
    progress_container = st.container()
    submit_button = st.button(label="To vector")
    if submit_button:
        load_qdrant()
        build_vector_store()


def save_html() -> None:
    """HTMLを保存する関数"""
    st.write("サイトを保存します。")
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(label="Message: ", key="input")
        submit_button = st.form_submit_button(label="Send")
    if submit_button and ".pdf" in user_input:
        content = requests.get(user_input, timeout=10).content
        save_path = "documents/pdf/" + "/".join(user_input.split("/")[2:])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)
    elif submit_button and user_input:
        content = requests.get(user_input, timeout=10).content
        save_path = "documents/html/" + "/".join(user_input.split("/")[2:]) + "index.html"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)


def chat_with_gpt() -> None:
    """GPTとのチャットを行う関数"""
    llm: ChatOpenAI = select_model()
    messages_container = st.container()
    input_container = st.container()
    place_input_form(input_container, llm)
    show_massages(messages_container)


def save_all_site() -> None:
    """指定されたURLのサイト全体を保存する関数"""
    st.write("サイト全体を保存します。")
    user_input, submit_button = form_input()

    if submit_button and user_input:
        cache = [user_input]
        searched_url_set: Set[str] = set()
        with st.spinner("サイトを保存しています。"):
            # stqdm はプログレスバーを表示するライブラリ
            # stadmを用いて実行状況を可視化したい。
            progress_bar = st.progress(0, text="Start")
            all_search_url_set = set()
            count = 0
            while cache:
                count += 1
                # all_search_url_setにcacheを追加
                all_search_url_set.update(cache)
                url = cache.pop()
                if process_url(url, user_input, cache, searched_url_set):
                    continue
                progress_bar.progress(
                    len(searched_url_set) / len(all_search_url_set),
                    text=f"実行状況: {len(searched_url_set)}/{len(all_search_url_set)} URL: {url} ",
                )

        display_saved_urls(searched_url_set)


def form_input() -> Tuple[str, bool]:
    """フォーム入力を処理する関数

    Returns:
        Tuple[str, bool]: ユーザー入力と送信ボタンの状態
    """
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(label="Message: ", key="input")
        submit_button = st.form_submit_button(label="Send")
    return user_input, submit_button


def process_url(url: str, user_input: str, cache: List[str], searched_url_set: Set[str]) -> bool:
    """URLを処理する関数

    Args:
        url (str): URL
        user_input (str): ユーザー入力
        cache (List[str]): キャッシュ
        searched_url_set (Set[str]): 検索済みのURLの集合

    Returns:
        bool: URLが処理済みかどうか
    """

    if url in searched_url_set or not url.startswith(user_input):
        return True

    searched_url_set.add(url)
    try:
        content = requests.get(url, timeout=10).content
        # ReadTimeout
    except requests.exceptions.RequestException as request_exception:
        print(f"Error occurred while processing URL {url}: {request_exception}")
        return True

    print(url)
    if content == b"":
        return True
    if url.endswith(".pdf"):
        save_pdf_content(url, user_input, content.decode("utf-8"))
    else:
        save_html_content(url, user_input, content.decode("utf-8"), cache)
    return False


def save_html_content(url: str, user_input: str, content: str, cache: List[str]) -> None:
    """
    HTMLのコンテンツを保存する関数

    Args:
        url (str): URL
        user_input (str): ユーザー入力
        content (str): HTMLのコンテンツ
        cache (List[str]): キャッシュ
    """

    save_path = generate_save_path(url, user_input, "html")
    create_and_write_file(save_path, content)

    try:
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a"):
            href = a.get("href")
            if href and (
                href.startswith("#")
                or href.startswith("mailto")
                or href.endswith(".jpg")
                or href.endswith(".png")
                or href.endswith(".gif")
            ):
                continue
            if href and href not in cache and href.startswith("http"):
                cache.append(href)
            else:
                cache.append(urljoin(url, href))
    except requests.exceptions.RequestException as request_exception:
        print(f"Error occurred while processing URL {url}: {request_exception}")
    return None


def save_pdf_content(url: str, user_input: str, content: str) -> None:
    """
    PDFのコンテンツを保存する関数

    Args:
        url (str): URL
        user_input (str): ユーザー入力
        content (str): PDFのコンテンツ

    Returns:
        None
    """

    save_path = generate_save_path(url, user_input, "pdf")
    create_and_write_file(save_path, content)
    return None


def generate_save_path(url: str, user_input: str, extension: str) -> str:
    """
    保存先のパスを生成する関数

    Args:
        url (str): URL
        user_input (str): ユーザー入力
        extension (str): 拡張子

    Returns:
        str: 保存先のパス
    """
    if extension == "html":
        if url == user_input:
            return "documents/html/" + "/".join(url.split("/")[2:]) + "index.html"
        elif url.endswith(".html"):
            return "documents/html/" + "/".join(url.split("/")[2:])
        else:
            return "documents/html/" + "/".join(url.split("/")[2:]) + "index.html"
    else:
        return "documents/pdf/" + "/".join(url.split("/")[2:])


def create_and_write_file(save_path: str, content: str) -> None:
    """
    ファイルを作成し、書き込む関数

    Args:
        save_path (str): 保存先のパス
        content (str): コンテンツ

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content.encode("utf-8"))


def display_saved_urls(searched_url_set: Set[str]) -> None:
    """保存したURLを表示する関数

    Args:
        searched_url_set (Set[str]): 検索済みのURLの集合
    """

    for url in searched_url_set:
        st.write(url)


def main() -> None:
    """メイン関数"""
    init()
    init_page()
    init_messages()
    selection = st.sidebar.radio("Go to", ["Save 1 page", "Save all sites", "Document to vector", "SiteChat"])
    if selection == "Save Documents":
        save_html()
    elif selection == "Save all site":
        save_all_site()
    elif selection == "Document to vector":
        document_to_vector()
    else:
        chat_with_gpt()


if __name__ == "__main__":
    main()
