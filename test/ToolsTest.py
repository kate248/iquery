from tools.AvailableFunctions import AvailableFunctions
from tools.Tools import *

af = AvailableFunctions(functions_list=[sql_inter, extract_data, python_inter])
print(af.functions)