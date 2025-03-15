
import openai
import os

from openai import  OpenAI
import inspect
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base="https://newone.nxykj.tech/v1"

client = OpenAI(api_key=openai.api_key ,base_url=openai.api_base)

class AvailableFunctions():
    """
    外部函数类，主要负责承接外部函数调用时相关功能支持。类属性包括外部函数列表、外部函数参数说明列表、以及调用方式说明三项。
    """

    def __init__(self, functions_list=[], functions=[], function_call="auto"):
        self.functions_list = functions_list
        self.functions = functions
        self.functions_dic = None
        self.function_call = None
        # 当外部函数列表不为空、且外部函数参数解释为空时，调用auto_functions创建外部函数解释列表
        if functions_list != []:
            self.functions_dic = {func.__name__: func for func in functions_list}
            self.function_call = function_call
            if functions == []:
                self.functions = auto_functions(functions_list)

    # 增加外部函数方法，并且同时可以更换外部函数调用规则
    def add_function(self, new_function, function_description=None, function_call_update=None):
        self.functions_list.append(new_function)
        self.functions_dic[new_function.__name__] = new_function
        if function_description == None:
            new_function_description = auto_functions([new_function])
            self.functions.append(new_function_description)
        else:
            self.functions.append(function_description)
        if function_call_update != None:
            self.function_call = function_call_update

def auto_functions(functions_list):
    """
    Chat模型的functions参数编写函数
    :param functions_list: 包含一个或者多个函数对象的列表；
    :return：满足Chat模型functions参数要求的functions对象
    """

    def functions_generate(functions_list):
        # 创建空列表，用于保存每个函数的描述字典
        functions = []
        # 对每个外部函数进行循环
        for function in functions_list:
            # 读取函数对象的函数说明
            function_description = inspect.getdoc(function)
            # 读取函数的函数名字符串
            function_name = function.__name__

            system_prompt = '以下是某的函数说明：%s' % function_description
            user_prompt = '根据这个函数的函数说明，请帮我创建一个JSON格式的字典，这个字典有如下5点要求：\
                               1.字典总共有三个键值对；\
                               2.第一个键值对的Key是字符串name，value是该函数的名字：%s，也是字符串；\
                               3.第二个键值对的Key是字符串description，value是该函数的函数的功能说明，也是字符串；\
                               4.第三个键值对的Key是字符串parameters，value是一个JSON Schema对象，用于说明该函数的参数输入规范。\
                               5.输出结果必须是一个JSON格式的字典，只输出这个字典即可，前后不需要任何前后修饰或说明的语句' % function_name

            response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            json_function_description = json.loads(
                    response.choices[0].message.content.replace("```", "").replace("json", ""))
            json_str = {"type": "function", "function": json_function_description}
            functions.append(json_str)
        return functions

    max_attempts = 4
    attempts = 0

    while attempts < max_attempts:
        try:
            functions = functions_generate(functions_list)
            break  # 如果代码成功执行，跳出循环
        except Exception as e:
            attempts += 1  # 增加尝试次数
            print("发生错误：", e)
            if attempts == max_attempts:
                print("已达到最大尝试次数，程序终止。")
                raise  # 重新引发最后一个异常
            else:
                print("正在重新运行...")
    return functions