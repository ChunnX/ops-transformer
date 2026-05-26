import sys
import openpyxl
def get_casename(file,start):
    wb = openpyxl.load_workbook(file, start)
    sheet = wb[op]
    data = sheet[start]
    case_name = data[0].value
    return case_name
file=sys.argv[1]
op=sys.argv[2]
strat=int(sys.argv[3])+1
print(get_casename(file,strat))
