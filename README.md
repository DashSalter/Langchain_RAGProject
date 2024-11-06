# Langchain_RAGProject

### python版本要求
```text
Python version==3.10
```
###


#### Model Url(需要下载的模型文件)
```text
链接：https://pan.baidu.com/s/1aiq6AS1ucAs_AX9-Wn_klQ?pwd=ARAR 
提取码：ARAR
```
###

#### 项目环境文件.env文件(openai_key)
需要在项目中创建.env文件，配置以下信息
```text
OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://api.openai.com/v1/"
```
###

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



