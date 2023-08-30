# src/app_manager.py

import os
import datetime
import dotenv
import openai
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.schema import SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant


class AppManager:
    """
    アプリケーションの管理を行うクラス。
    """

    def __init__(self) -> None:
        dotenv.load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def load_qdrant(self) -> Qdrant:
        """Qdrantをロードする関数。

        Returns:
            Qdrant: Qdrantのインスタンス
        """
        qdrant_path: str = str(os.getenv("QDRANT_PATH"))
        collection_name: str = str(os.getenv("COLLECTION_NAME"))
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

    def init_page(self) -> None:
        """Streamlitページの初期設定を行う関数"""
        st.set_page_config(page_title="ChatGPT", page_icon="🤗")
        st.header("ChatGPT 🤗")
        st.sidebar.title("Options")

    def init_messages(self) -> None:
        """メッセージの初期設定を行う関数"""
        init_content = f"""
        You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}.
        """
        clear_button = st.sidebar.button("Clear Conversation", key="clear")
        if clear_button or "messages" not in st.session_state:
            st.session_state.messages = [SystemMessage(content=init_content)]
            st.session_state.costs = []
