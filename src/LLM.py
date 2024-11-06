from typing import Iterable

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Output
from langchain_openai import ChatOpenAI

from src import Knowledge


class LLM:
    _chat_history = ChatMessageHistory()  # 对话历史

    def __init__(self, knowledge: Knowledge = None, chat_history_max_length: int = 8):

        self.knowledge: Knowledge = knowledge
        self.chat_history_max_length: int = chat_history_max_length

        self.knowledge_prompt = None  # 问答模板
        self.normal_prompt = None  # 正常模板
        self.create_chat_prompt()  # 创建聊天模板

    def create_chat_prompt(self) -> None:
        ai_info = '你叫瓜皮，一个帮助人们解答各种问题的助手。'

        # AI系统prompt
        knowledge_system_prompt = (
            f"{ai_info} 使用检索到的上下文来回答问题。如果你不知道答案，就说你不知道。\n\n"
            "{context}"
        )

        self.knowledge_prompt = ChatPromptTemplate.from_messages(  # 知识库prompt
            [
                ("system", knowledge_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # 没有指定知识库的模板的AI系统模板
        self.normal_prompt = ChatPromptTemplate.from_messages(  # 正常prompt
            [
                ("system", ai_info),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

    @staticmethod
    def streaming_parse(chunks: Iterable[AIMessageChunk]) -> list[AddableDict]:
        """统一模型的输出格式，将模型的输出存储到字典answer的value中"""
        for chunk in chunks:
            yield AddableDict({'answer': chunk.content})

    def get_chain(self, collection: str, model: str, max_length: int, temperature: float) -> RunnableWithMessageHistory:
        """获取聊天链"""
        chat = ChatOpenAI(model=model, max_tokens=max_length, temperature=temperature)

        if collection is None:
            rag_chain = self.normal_prompt | chat | self.streaming_parse
        else:
            _retriever = self.knowledge.get_retrievers(collection)  # 构建检索器

            # 只保留指定的知识库个记录
            if len(self._chat_history.messages) > self.chat_history_max_length:
                self._chat_history.messages = self._chat_history.messages[-self.chat_history_max_length:]

            # 构建问答链
            question_answer_chain = create_stuff_documents_chain(chat, self.knowledge_prompt)
            rag_chain = create_retrieval_chain(_retriever, question_answer_chain)

        return RunnableWithMessageHistory(
            rag_chain,  # 传入聊天链
            lambda session_id: self._chat_history,  # 传入历史信息
            input_messages_key="input",  # 输入信息的键名
            history_messages_key="chat_history",  # 历史信息的键名
            output_messages_key="answer",  # 输出答案
        )

    def invoke(self, question: str, collection: str = '', model="gpt-3.5-turbo",
               max_length=256, temperature=1) -> Output:
        """
        :param question: 用户提出的问题 例如: '请问你是谁？'
        :param collection: 知识库文件名称 例如:'人事管理流程.docx'
        :param model: 使用模型,默认为 'gpt-3.5-turbo'
        :param max_length: 数据返回最大长度
        :param temperature: 数据温度值
        """
        return self.get_chain(
            collection, model, max_length, temperature
        ).invoke(
            {"input": question}, {"configurable": {"session_id": "unused"}}
        )

    def clear_history(self) -> None:
        """清除历史信息"""
        self._chat_history.clear()

    def get_history_message(self) -> list:
        """获取历史信息"""
        return self._chat_history.messages
