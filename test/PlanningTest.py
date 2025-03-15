from planning.Planning import *
from memory.MessageManager import MessageManager
from tools.AvailableFunctions import AvailableFunctions
from tools.Tools import *


# 数据字典文件
with open('D:\yuanma\iquery\iquery知识库\iquery数据字典.md', 'r', encoding='utf-8') as f:
    data_dictionary = f.read()

# msg1 = MessageManager(system_content_list=[data_dictionary], question="请帮我简单介绍iquery数据库基本情况。")
#
# msg_response1 = get_chat_response(model='gpt-3.5-turbo',
#                                   messages=msg1)

af = AvailableFunctions(functions_list=[sql_inter, extract_data, python_inter])
print(af.functions)


msg3 = MessageManager(system_content_list=[data_dictionary], question="请帮我统计user_demographics总共有多少条数据？")

msg_response3 = get_chat_response(model='gpt-3.5-turbo-16k',
                                  messages=msg3,
                                  available_functions=af)