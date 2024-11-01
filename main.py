import os

import gradio as gr

from src import LLM, Knowledge

# 代理设置，确保不会出现链接异常
proxy_url = 'http://127.0.0.1'
proxy_port = '7890'  # 代理的端口
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'


class GUI:
    knowledge = Knowledge(reorder=False)  # 实例化知识库 reorder=False表示不对检索结果进行排序,因为太占用时间了
    llm = LLM(knowledge, chat_history_max_length=10)  # 实例化LLM模型
    # 定义一个包含两个LLM模型名称的列表，供用户选择
    llm_models = ["gpt-3.5-turbo", "gpt-4o"]

    def __init__(self):
        self.create_gui()

    def create_gui(self):
        # 创建一个Gradio Blocks应用，设置fill_height为True
        with gr.Blocks(fill_height=True) as self.demo:
            # 创建一个新的行布局
            with gr.Row():
                # 创建一个占比为 4 的列布局
                with gr.Column(scale=4):
                    # 创建一个下拉菜单，用于选择LLM模型
                    model = gr.Dropdown(
                        choices=self.llm_models,
                        value=self.llm_models[0],
                        label="LLM Model",
                        interactive=True,
                        scale=1,
                    )
                    # 创建一个聊天机器人界面
                    chatbot = gr.Chatbot(show_label=False, scale=3, show_copy_button=True, type="messages")

                # 创建一个占比为 1 的列布局，显示进度
                with gr.Column(scale=1, show_progress=True):
                    # 创建一个滑块，用于设置生成回复的最大长度
                    max_length = gr.Slider(1, 4095, value=250, step=1.0, label="Maximum length", interactive=True)
                    # 创建一个滑块，用于设置生成回复的温度
                    temperature = gr.Slider(0, 2, value=1, step=0.01, label="Temperature", interactive=True)
                    # 创建一个按钮，用于清除聊天记录
                    clear = gr.Button("清除")
                    # 创建一个下拉菜单，用于选择知识库
                    collection = gr.Dropdown(choices=self.knowledge.get_document_list(), label="Knowledge")
                    # 创建一个文件上传控件，支持多种文件类型
                    file = gr.File(label="上传文件", file_types=['.doc', '.docx', '.csv', '.txt', '.pdf', '.md'])

            # 创建一个文本框，用于用户输入
            user_input = gr.Textbox(placeholder="Input...", show_label=False)
            # 创建一个按钮，用于提交用户输入
            user_submit = gr.Button("提交")

            # 绑定 clear 按钮的点击事件，清除模型历史记录，并更新聊天机器人界面
            clear.click(fn=self.llm.clear_history, inputs=None, outputs=[chatbot])

            user_input.submit(
                fn=self.submit,
                inputs=[user_input, chatbot],  # 一个是用户输入 一个当前的聊天记录
                outputs=[user_input, chatbot]  # 一个是为了清空用户输入的文本框 另一个是更新后的聊天记录，将新的用户查询添加到聊天记录中。
            ).then(
                fn=self.llm_reply,
                inputs=[collection, chatbot, model, max_length, temperature],

                outputs=[chatbot]
                # 更新后的聊天记录，将模型生成的回复添加到聊天记录中。
            )
            # 绑定用户输入文本框的提交事件，
            # 先调用submit函数，
            # 然后调用llm_reply函数，
            # 并更新聊天机器人界面
            user_submit.click(
                fn=self.submit,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot]
            ).then(
                fn=self.llm_reply,
                inputs=[collection, chatbot, model, max_length, temperature],
                outputs=[chatbot]
            )
            # 绑定提交按钮的点击事件，先调用submit函数，
            # 然后调用llm_reply函数，并更新聊天机器人界面

            # 绑定文件上传控件的上传事件，调用upload_knowledge函数，并更新文件控件和知识库下拉菜单
            file.upload(
                fn=self.knowledge.upload_knowledge,
                inputs=[file]
            ).then(
                fn=self.update_dropdown,
                outputs=[collection]
            )

            # 绑定知识库下拉菜单的更改事件，调用clear_history函数，并更新聊天机器人界面 也就是换一个知识库就清空当前的页面
            collection.change(fn=self.llm.clear_history, inputs=None, outputs=[chatbot])

            # 绑定应用加载事件，调用clear_history函数，并更新聊天机器人界面
            self.demo.load(fn=self.llm.clear_history, inputs=None, outputs=[chatbot])

    @staticmethod
    def submit(query: str, chat_history: list) -> tuple[str, list]:
        """用于处理用户提交的查询 空的字符用于清空输入框数据"""
        query = query.strip()
        if query == '': return '', chat_history  # 如果问题为空则不做处理
        chat_history.append({"role": "user", "content": query})  # 将问题添加到聊天记录中
        return '', chat_history

    def llm_reply(self, collection_: str, chat_history: list, model_: str,
                  max_length_: int = 256, temperature_: float = 1) -> list:
        """定义llm_reply函数，用于生成模型回复
        # collection: 用户选择的知识库。
        # chat_history: 当前的聊天记录（已经包含用户的新查询）。
        # model: 用户选择的 LLM 模型。
        # max_length: 用户设置的生成回复的最大长度。
        # temperature: 用户设置的生成回复的温度。
            低 temperature：适合需要准确性和一致性的任务，比如回答事实性问题或撰写正式文档。
            高 temperature：适合需要创造性和多样性的任务，比如生成诗歌、故事或进行头脑风暴。
        """
        print(f'使用知识库:{collection_}')
        question = chat_history[-1]['content']

        response = self.llm.invoke(  # 使用LLM模型生成回复
            question, collection_, model=model_,
            max_length=max_length_, temperature=temperature_
        )
        chat_history.append({"role": "assistant", "content": response['answer']})
        return chat_history

    # 定义更新下拉框的函数
    def update_dropdown(self) -> dict:
        # 更新下拉框的选项为最新的知识库列表
        return gr.update(choices=self.knowledge.get_document_list())

    def run(self) -> None:
        self.demo.launch(share=True)


if __name__ == '__main__':
    GUI().run()
