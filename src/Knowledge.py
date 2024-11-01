import json
import os
from hashlib import md5

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAI

from src import DocumentChunker

# 设置知识库 向量模型 重排序模型的路径
embedding_model = 'BAAI/bge-large-zh-v1.5'  # 向量模型
rerank_model = 'BAAI/bge-reranker-large'  # 重排序模型
chroma_dir = 'chroma/'  # 向量数据库的路径
# 向量模型参数,cpu表示使用cpu进行计算，gpu表示使用gpu进行计算
model_kwargs = {'device': 'cpu'}
vector_document = 'vector_document.json'  # 记录向量化的文档


class Knowledge:
    """知识库"""
    # 向量化模型,使用的本地模型
    _embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)
    _llm = OpenAI(temperature=0)

    def __init__(self, reorder=False):
        self.reorder = reorder  # 是否重排序 启动重排序模型，时间会增加
        self.vector_document = {}

    @staticmethod
    def is_already_vector_database(collection_name: str) -> bool:
        """是否已经对文件进行向量存储"""
        return True if os.path.exists(os.path.join(chroma_dir, collection_name)) else False

    def upload_knowledge(self, temp_file):
        """上传知识库文件"""
        file_path = os.path.basename(temp_file)
        # 当前文件的md5值作为表面，随便判断是否已经存在向量数据库
        collection_name = self.get_file_md5(file_path)
        if self.is_already_vector_database(collection_name):
            print('该文件已经存在，不再上传！')
            return False
        else:
            self.create_indexes(collection_name, file_path)

    def load_knowledge(self, collection_name) -> Chroma:
        """加载向量数据库"""
        persist_directory = os.path.join('./chroma', collection_name)
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self._embeddings
        )

    def get_retrievers(self, collection: str) -> BaseRetriever:
        """根据文档名称获取该文档的检索器"""
        file_path = os.path.basename(collection)
        # 当前文件的md5值作为表面，随便判断是否已经存在向量数据库
        collection_name = self.get_file_md5(file_path)

        if self.is_already_vector_database(collection_name):
            retriever = self.load_knowledge(collection_name).as_retriever(search_kwargs={'k': 3})
            if self.reorder:
                return self.contex_reorder(retriever)
            else:
                return retriever

    @staticmethod
    def contex_reorder(retriever) -> ContextualCompressionRetriever:
        """交叉编码器重新排序器"""
        # 加载重新排序的模型
        model = HuggingFaceCrossEncoder(model_name=rerank_model, model_kwargs=model_kwargs)
        compressor = CrossEncoderReranker(model=model, top_n=3)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever

    def create_indexes(self, collection_name: str, file_path: str) -> None:
        """将段落数据向量化后添加到向量数据库"""

        print(f'开始向量化文件 {file_path}...')
        loader = DocumentChunker(file_path)
        documents = loader.load()  # 加载段落
        print('文档段落加载完成！')
        print(f'开始向量化文档...')

        # 创建存储向量数据库的目录
        persist_directory = os.path.join('./chroma', collection_name)
        Chroma.from_documents(documents, persist_directory=persist_directory,
                              collection_name=collection_name, embedding=self._embeddings)

        # 记录向量化的文档
        self.create_vector_document(file_path, collection_name)
        print(f'向量化文件 {file_path} 完成！')

    def create_vector_document(self, file_path: str, collection_name: str) -> None:
        """构建文档"""
        data = self.get_vector_document_name()
        data.update({file_path: collection_name})
        with open(vector_document, 'w', encoding='utf-8') as file:
            file.write(json.dumps(data, ensure_ascii=False, indent=4))

    @staticmethod
    def get_vector_document_name() -> dict:
        try:
            with open(vector_document, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get_document_list(self) -> list:
        """获取已向量化的文档列表"""
        return list(self.get_vector_document_name().keys())

    @staticmethod
    def get_file_md5(file_path: str) -> str:
        """对文件中的内容计算md5值"""
        try:
            block_size = 65536  # 每次读取的块大小
            m = md5()  # 创建MD5对象
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(block_size)
                    if not data:
                        break
                    m.update(data)  # 更新MD5值
            return m.hexdigest()  # 返回计算结果的十六进制字符串格式
        except FileNotFoundError:
            raise FileNotFoundError('文件没有找到!')
        except Exception as e:
            raise Exception(f"计算MD5值时出现异常: {e}")
