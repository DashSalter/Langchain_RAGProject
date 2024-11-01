from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader)
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.file_utils.filetype import FileType, detect_filetype

"""
detect_filetype 函数中的 361行加上以下代码
if LIBMAGIC_AVAILABLE:
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
"""


class DocumentChunker(BaseLoader):
    """文档加载与切分"""
    allow_file_type = {  # 文件类型与加载类及参数
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True, "encoding": "utf-8"}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True, "encoding": "utf-8"}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {"encoding": "utf-8"}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {"encoding": "utf-8"}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {"encoding": "utf-8"})
    }

    def __init__(self, file_path: str) -> None:
        """加载文档对象和分割器"""
        self.file_path = file_path
        self.file_type_ = detect_filetype(file_path)
        if self.file_type_ not in self.allow_file_type:
            raise ValueError(f"Unsupported file type: {self.file_type_}")

        # 获取文件的类型和参数
        loader_class, params = self.allow_file_type[self.file_type_]
        # 在类中传入文件路径和参数,创建加载类示例对象
        self.loader: BaseLoader = loader_class(file_path, **params)
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def load(self) -> list:
        """使用文本切割器对文件进行切分"""
        return self.loader.load_and_split(self.text_splitter)

    def __str__(self) -> str:
        return f"<DocumentChunker source:{self.file_path} file_type:{self.file_type_}>"


if __name__ == '__main__':
    file_path1 = "../人事管理流程.docx"  # test docx file
    chunker = DocumentChunker(file_path1)
    chunks = chunker.load()
    print(len(chunks))
