## 导入依赖
import openai
import tiktoken

import copy

class MessageManager():
    """
    MessageManager类，用于创建Chat模型能够接收和解读的messages对象。该对象是原始Chat模型接收的\
    messages对象的更高级表现形式，MessageManager类对象将字典类型的list作为其属性之一，同时还能\
    能区分系统消息和历史对话消息，并且能够自行计算当前对话的token量，并执能够在append的同时删\
    减最早对话消息，从而能够更加顺畅的输入大模型并完成多轮对话需求。
    """

    def __init__(self,
                 system_content_list=[],
                 question='你好。',
                 tokens_thr=None,
                 project=None):

        self.system_content_list = system_content_list
        # 系统消息文档列表，相当于外部输入文档列表
        system_messages = []
        # 除系统消息外历史对话消息
        history_messages = []
        # 用于保存全部消息的list
        messages_all = []
        # 系统消息字符串
        system_content = ''
        # 历史消息字符串，此时为用户输入信息
        history_content = question
        # 系统消息+历史消息字符串
        content_all = ''
        # 输入到messages中系统消息个数，初始情况为0
        num_of_system_messages = 0
        # 全部信息的token数量
        all_tokens_count = 0

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # 将外部输入文档列表依次保存为系统消息
        if system_content_list != []:
            for content in system_content_list:
                system_messages.append({"role": "system", "content": content})
                # 同时进行全文档拼接
                system_content += content

            # 计算系统消息token
            system_tokens_count = len(encoding.encode(system_content))
            # 拼接系统消息
            messages_all += system_messages
            # 计算系统消息个数
            num_of_system_messages = len(system_content_list)

            # 若存在最大token数量限制
            if tokens_thr != None:
                # 若系统消息超出限制
                if system_tokens_count >= tokens_thr:
                    print("system_messages的tokens数量超出限制，当前系统消息将不会被输入模型")
                    # 删除系统消息
                    system_messages = []
                    messages_all = []
                    # 系统消息个数清零
                    num_of_system_messages = 0
                    # 系统消息token数清零
                    system_tokens_count = 0

            all_tokens_count += system_tokens_count

        # 创建首次对话消息
        history_messages = [{"role": "user", "content": question}]
        # 创建全部消息列表
        messages_all += history_messages

        # 计算用户问题token
        user_tokens_count = len(encoding.encode(question))

        # 计算总token数
        all_tokens_count += user_tokens_count

        # 若存在最大token限制
        if tokens_thr != None:
            # 若超出最大token限制
            if all_tokens_count >= tokens_thr:
                print("当前用户问题的tokens数量超出限制，该消息无法被输入到模型中")
                # 同时清空系统消息和用户消息
                history_messages = []
                system_messages = []
                messages_all = []
                num_of_system_messages = 0
                all_tokens_count = 0

        # 全部messages信息
        self.messages = messages_all
        # system_messages信息
        self.system_messages = system_messages
        # user_messages信息
        self.history_messages = history_messages
        # messages信息中全部content的token数量
        self.tokens_count = all_tokens_count
        # 系统信息数量
        self.num_of_system_messages = num_of_system_messages
        # 最大token数量阈值
        self.tokens_thr = tokens_thr
        # token数计算编码方式
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # message挂靠的项目
        self.project = project

    # 删除部分对话信息
    def messages_pop(self, manual=False, index=None):
        def reduce_tokens(index):
            drop_message = self.history_messages.pop(index)
            self.tokens_count -= len(self.encoding.encode(str(drop_message)))

        if self.tokens_thr is not None:
            while self.tokens_count >= self.tokens_thr:
                reduce_tokens(-1)

        if manual:
            if index is None:
                reduce_tokens(-1)
            elif 0 <= index < len(self.history_messages) or index == -1:
                reduce_tokens(index)
            else:
                raise ValueError("Invalid index value: {}".format(index))

        # 更新messages
        self.messages = self.system_messages + self.history_messages

    # 增加部分对话信息
    def messages_append(self, new_messages):

        # 若是单独一个字典，或JSON格式字典
        if type(new_messages) is dict or type(
                new_messages) is openai.types.chat.chat_completion_message.ChatCompletionMessage:
            self.messages.append(new_messages)
            self.tokens_count += len(self.encoding.encode(str(new_messages)))

        # 若新消息也是MessageManager对象
        elif isinstance(new_messages, MessageManager):
            self.messages += new_messages.messages
            self.tokens_count += new_messages.tokens_count

        # 重新更新history_messages
        self.history_messages = self.messages[self.num_of_system_messages:]

        # 再执行pop，若有需要，则会删除部分历史消息
        self.messages_pop()

    # 复制信息
    def copy(self):
        # 创建一个新的 MessageManager 对象，复制所有重要的属性
        system_content_str_list = [message["content"] for message in self.system_messages]
        new_obj = MessageManager(
            system_content_list=copy.deepcopy(system_content_str_list),  # 使用深复制来复制系统消息
            question=self.history_messages[0]["content"] if self.history_messages else '',
            tokens_thr=self.tokens_thr
        )
        # 复制任何其他需要复制的属性
        new_obj.history_messages = copy.deepcopy(self.history_messages)  # 使用深复制来复制历史消息
        new_obj.messages = copy.deepcopy(self.messages)  # 使用深复制来复制所有消息
        new_obj.tokens_count = self.tokens_count
        new_obj.num_of_system_messages = self.num_of_system_messages

        return new_obj

    # 增加系统消息
    def add_system_messages(self, new_system_content):
        system_content_list = self.system_content_list
        system_messages = []
        # 若是字符串，则将其转化为list
        if type(new_system_content) == str:
            new_system_content = [new_system_content]

        system_content_list.extend(new_system_content)
        new_system_content_str = ''
        for content in new_system_content:
            new_system_content_str += content
        new_token_count = len(self.encoding.encode(str(new_system_content_str)))
        self.tokens_count += new_token_count
        self.system_content_list = system_content_list
        for message in system_content_list:
            system_messages.append({"role": "system", "content": message})
        self.system_messages = system_messages
        self.num_of_system_messages = len(system_content_list)
        self.messages = system_messages + self.history_messages

        # 再执行pop，若有需要，则会删除部分历史消息
        self.messages_pop()

    # 删除系统消息
    def delete_system_messages(self):
        system_content_list = self.system_content_list
        if system_content_list != []:
            system_content_str = ''
            for content in system_content_list:
                system_content_str += content
            delete_token_count = len(self.encoding.encode(str(system_content_str)))
            self.tokens_count -= delete_token_count
            self.num_of_system_messages = 0
            self.system_content_list = []
            self.system_messages = []
            self.messages = self.history_messages

    # 清除对话消息中的function消息
    def delete_function_messages(self):
        # 用于删除外部函数消息
        history_messages = self.history_messages
        # 从后向前迭代列表
        for index in range(len(history_messages) - 1, -1, -1):
            message = history_messages[index]
            ## 这儿估计有问题
            if message.get("function_call") or message.get("role") == "function":
                self.messages_pop(manual=True, index=index)
