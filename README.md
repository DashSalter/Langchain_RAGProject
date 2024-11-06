# Langchain_RAGProject

Python version==3.10

Model url
链接：https://pan.baidu.com/s/1aiq6AS1ucAs_AX9-Wn_klQ?pwd=ARAR 
提取码：ARAR



会出现的异常提示解决方案
```python
The MIME type of 'C:\\Users\\人事管理流程.docx' is "cannot open `c:\\users\\\\344\\272\\272\\344\\272\\213\\347\\256\\241\\347\\220\\206\\346\\265\\201\\347\\250\\213.docx' (no such file or directory)". This file type is not currently supported in unstructured.
```

解决方案
"""
```python
from unstructured.file_utils.filetype import FileType, detect_filetype
#detect_filetype 函数中的 361行加上以下代码
if LIBMAGIC_AVAILABLE:
    import magic

    mime_type = (

# 修改成以下代码
if LIBMAGIC_AVAILABLE:
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    import magic

    mime_type = (
    
```



