import pandas
import os
import csv
import sys
import openpyxl
def init_csv(file):
    item=['op_name', 'case_name', 'timestamp', 'precision', 'result','errorcode','time_cost', 'bound_type', 'block_dim', ' max_deviation', 'input_shape', 'input_dtype', 'input_range', 'cycle_list', 'task_type', 'e2e_time', 'Average_workspace', 'Average_aclnn', 'Average_sync', 'workspacesize', 'cost memory', 'workspacesize(golden)', 'cost memory(golden)', 'mem compare', 'duration', 'wait time', 'duration(golden)', 'wait time(golden)', 'aclprof compare']
    if os.path.exists(file):
        os.remove(file)
    with open(file, mode="a", encoding='utf-8-sig') as f:
        write = csv.writer(f)
        write.writerow(item)

def write_csv(file,data):
    with open(file,mode="a",encoding='utf-8') as f:
        write=csv.writer(f)
        write.writerow(data)


def get_op_name(file,start,end):
    wb = openpyxl.load_workbook(file)
    sheet = wb[op]
    case_name_list = []
    op_diction={}
    for index in range(int(start), int(end) + 1):
        data = sheet[index]
        case_name = data[0].value
        op_name = data[-1].value
        op_diction[case_name]=op_name
        # print(op_name, case_name)
        case_name_list.append(case_name)
    return op_diction



def get_casename(file,start,end):
    wb = openpyxl.load_workbook(file)
    sheet = wb[op]
    case_name_list=[]
    for index in range(int(start),int(end)+1):
        data = sheet[index]
        case_name = data[0].value
        op_name=data[-1].value
        print(op_name,case_name)
        case_name_list.append(case_name)
    return case_name_list

result_list = []
def get_list(case_list):
    path = os.getcwd()
    file_list = []
    result_case_list = []
    if not os.path.exists("MC2_result.csv"):
        init_csv("MC2_result.csv")
    for file in os.listdir(os.getcwd()):
         if file.startswith("allreduce_a8w8_result")or file.startswith("allreduce_result")  or file.startswith("reducescatterv2_result") or file.startswith(
                "allgatherv2_result") or file.startswith("allreduce_a16w8_result") or file.startswith("allreduce_a16w4_result") :
            file_list.append(file)
    for precision_file in file_list:
        # if not os.path.exists("MC2_result.csv"):
        #     init_csv("MC2_result.csv")
        if precision_file.startswith("allgather") or precision_file.startswith("mm_allreduce_ARN_result") :
            data = pandas.read_csv(precision_file, names=["case_name", "rank", "result", "gather"])
        else:
            data = pandas.read_csv(precision_file, names=["case_name", "rank", "result"])
        list_2 = data.values.tolist()
        case_name_list = set()
        for s in list_2:
            case_name_list.add(s[0])
        case_name_list = list(case_name_list)
        for case_name in case_name_list:
            result_case_list.append(case_name)
            pass_count = 0
            for num in list_2:
                if num[0] == case_name:
                    try:
                        if  "PASS" in num[2] or "Pass" in num[2]:
                            pass_count += 1
                    except Exception as  e:
                        print(e)

            if pass_count == 8 or pass_count == 16 or pass_count==4 or pass_count==2:
                precision = "PASS"
                result_list.append([op_name[case_name], case_name, '', 100, "Pass"])
            else:
                precision = "FAIL"
                result_list.append([op_name[case_name], case_name, '', 100, "Failed"])
    case_list = [s for s in case_list if s not in result_case_list]
    for s in case_list:
        result_list.append([op_name[s], s,'', 99, "Failed", "None"])
    return result_list



if __name__ == '__main__':
    file = sys.argv[1]
    op = sys.argv[2]
    strat = int(sys.argv[3]) + 1
    end = int(sys.argv[4]) + 1
    case_list=get_casename(file,strat,end)
    op_name=get_op_name(file,strat,end)
    get_list(case_list)
    for result in result_list:
            write_csv("MC2_result.csv", result)


