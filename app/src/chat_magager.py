# src/chat_manager.py

from typing import Any
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage


class ChatManager:
    def __init__(self) -> None:
        pass

    def select_model(self) -> ChatOpenAI:
        model = st.sidebar.radio("Choose a model:", ("GPT-3.5-16k", "GPT-4"))
        if model == "GPT-3.5":
            st.session_state.model_name = "gpt-3.5-turbo-16k"
        else:
            st.session_state.model_name = "gpt-4"
        st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
        return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

    def show_messages(self, messages_container: Any) -> None:
        messages = st.session_state.get("messages", [])
        with messages_container:
            for message in messages:
                if isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                else:
                    st.write(f"System message: {message.content}")

    def place_input_form(self, input_container: Any, llm: ChatOpenAI) -> None:
        with input_container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area(label="Message: ", key="input")
                submit_button = st.form_submit_button(label="Send")
            if submit_button and user_input:
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("ChatGPT is typing ..."):
                    response = build_qa_model(llm).run(user_input)
                st.session_state.messages.append(AIMessage(content=response))

    def build_qa_model(self, llm: ChatOpenAI) -> Any:
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
