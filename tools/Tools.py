import  json


import pymysql
import pandas as pd






def sql_inter(sql_query, g='globals()'):
    """
    用于获取iquery数据库中各张表的有关相关信息，\
    核心功能是将输入的SQL代码传输至iquery数据库所在的MySQL环境中进行运行，\
    并最终返回SQL代码运行结果。需要注意的是，本函数是借助pymysql来连接MySQL数据库。
    :param sql_query: 字符串形式的SQL查询语句，用于执行对MySQL中iquery数据库中各张表进行查询，并获得各表中的各类相关信息
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：sql_query在MySQL中的运行结果。
    """

    mysql_pw = "iquery_agent"

    connection = pymysql.connect(
        host='localhost',  # 数据库地址
        user='iquery_agent',  # 数据库用户名
        passwd=mysql_pw,  # 数据库密码
        db='iquery',  # 数据库名
        charset='utf8'  # 字符集选择utf8
    )

    try:
        with connection.cursor() as cursor:
            # SQL查询语句
            sql = sql_query
            cursor.execute(sql)

            # 获取查询结果
            results = cursor.fetchall()

    finally:
        connection.close()

    return json.dumps(results)


def extract_data(sql_query, df_name, g='globals()'):
    """
    用于借助pymysql，将MySQL中的iquery数据库中的表读取并保存到本地Python环境中。
    :param sql_query: 字符串形式的SQL查询语句，用于提取MySQL中iquery数据库中的某张表。
    :param df_name: 将MySQL数据库中提取的表格进行本地保存时的变量名，以字符串形式表示。
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：表格读取和保存结果
    """

    mysql_pw = "iquery_agent"

    connection = pymysql.connect(
        host='localhost',  # 数据库地址
        user='iquery_agent',  # 数据库用户名
        passwd=mysql_pw,  # 数据库密码
        db='iquery',  # 数据库名
        charset='utf8'  # 字符集选择utf8
    )

def python_inter(py_code, g='globals()'):
    """
    用于对iquery数据库中各张数据表进行查询和处理，并获取最终查询或处理结果。
    :param py_code: 字符串形式的Python代码，用于执行对iquery数据库中各张数据表进行操作
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    :return：代码运行的最终结果
    """
    # 添加图片对象，如果存在绘图代码，则创建fig对象
    py_code = insert_fig_object(py_code)
    global_vars_before = set(globals().keys())
    try:
        exec(py_code, globals())
    except Exception as e:
        return str(e)
    global_vars_after = set(globals().keys())
    new_vars = global_vars_after - global_vars_before
    if new_vars:
        result = {var: globals()[var] for var in new_vars}
        return str(result)
    else:
        try:
            return str(eval(py_code, globals()))
        except Exception as e:
            return "已经顺利执行代码"

    globals()[df_name] = pd.read_sql(sql_query, connection)

    return "已成功完成%s变量创建" % df_name


def insert_fig_object(code_str, g='globals()'):
    """
    为图片创建fig对象
    :param g: g，字符串形式变量，表示环境变量，无需设置，保持默认参数即可
    """
    print("开始画图了")
    global fig
    # 检查是否已存在 fig 对象的创建
    if 'fig = plt.figure' in code_str or 'fig, ax = plt.subplots()' in code_str:
        return code_str  # 如果存在，则返回原始代码字符串

    # 定义可能的库别名和全名
    plot_aliases = ['plt.', 'matplotlib.pyplot.', 'plot']
    sns_aliases = ['sns.', 'seaborn.']

    # 寻找第一次出现绘图相关代码的位置
    first_plot_occurrence = min(
        (code_str.find(alias) for alias in plot_aliases + sns_aliases if code_str.find(alias) >= 0), default=-1)

    # 如果找到绘图代码，则在该位置之前插入 fig 对象的创建
    if first_plot_occurrence != -1:
        plt_figure_index = code_str.find('plt.figure')
        if plt_figure_index != -1:
            # 寻找 plt.figure 后的括号位置，以确定是否有参数
            closing_bracket_index = code_str.find(')', plt_figure_index)
            # 如果找到了 plt.figure()，则替换为 fig = plt.figure()
            modified_str = code_str[:plt_figure_index] + 'fig = ' + code_str[
                                                                    plt_figure_index:closing_bracket_index + 1] + code_str[
                                                                                                                  closing_bracket_index + 1:]
        else:
            modified_str = code_str[:first_plot_occurrence] + 'fig = plt.figure()\n' + code_str[first_plot_occurrence:]
        return modified_str
    else:
        return code_str  # 如果没有找到绘图代码，则返回原始代码字符串

