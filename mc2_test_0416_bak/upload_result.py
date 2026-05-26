import random
import shutil

import pandas
import os
import csv
import sys
import openpyxl
import subprocess
from datetime import datetime
import os
import pymysql
import codecs
import socket

command1="cat /usr/local/Ascend/latest/compiler/version.info |grep timestamp"
version="None"
try:
    timestamp=subprocess.check_output(command1, shell=True).decode('utf-8').replace("timestamp=","")
except Exception as e:
    timestamp="None"
version="{}_newest".format(timestamp.strip())
print(version)
now = datetime.now()
formatted_time = str(now.strftime("%Y-%m-%d %H:%M:%S"))

def readfile(file):
    a = []
    for line in open(file):
        case_name = line.strip("\n").split(",")[0]
        if case_name not in a:
            a.append(case_name)
    return a



def init_csv(file):
    item=['用例名', '组件', '任务ID', 'version', '状态','开始时间','结束时间', '用例总数', '执行成功数', '执行失败数']
    if os.path.exists(file):
        os.remove(file)
    with open(file, mode="a", encoding='utf-8-sig') as f:
        write = csv.writer(f)
        write.writerow(item)

def write_csv(file,data):
    with open(file,mode="a",encoding='utf-8') as f:
        write=csv.writer(f)
        write.writerow(data)

def query_112(sql):
    conn = pymysql.connect(host="100.102.207.234", user="root", password="root", database="testDB")
    cursor = conn.cursor()
    cursor.execute(sql)
    result_all = cursor.fetchall()
    conn.commit()
    conn.close()
    cursor.close()
    if result_all != ():
        result = list(result_all)
    else:
        result = []
    return result


def insert_112(sql, data_list):
    conn = pymysql.connect(host="100.102.207.234", user="root", password="root", database="testDB")
    cursor = conn.cursor()
    cursor.executemany(sql, data_list)
    conn.commit()
    conn.close()
    cursor.close()


def get_mc2_data(taskid, network, report, ip, env_type):
    data_list = []
    data_detail_list = []
    component, version, start_time, finish_time, status = '', '', '', '', ''
    total_num_total, success_num_total, fail_num_total = 0, 0, 0
    with codecs.open(report, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            case_name = row['用例名']
            component = row['组件']
            version = row['version']
            status = row['状态']
            start_time = row['开始时间']
            finish_time = row['结束时间']
            total_num = int(row['用例总数'])
            success_num = int(row['执行成功数'])
            fail_num = int(row['执行失败数'])
            result = 'Fail' if fail_num else 'Pass'
            total_num_total += total_num
            success_num_total += success_num
            fail_num_total += fail_num
            # print([taskid, case_name, finish_time, version, component, env_type, result])
            data_detail_list.append([taskid, case_name, finish_time, version, component, env_type, result])

    x = success_num_total + fail_num_total
    y = '%.3f' % (x / total_num_total) if total_num_total else 0
    z = '%.3f' % (success_num_total / x) if x else 0

    if x < total_num_total and y == 1.000:
        y = 0.999
    if success_num_total < x and z == 1.000:
        z = 0.999
    presure = total_num_total
    if "fuzz" in network:
        presure=total_num_total
    else:
        presure = total_num_total*2
    data_list.append([component, network, taskid, status, start_time, finish_time, total_num_total, success_num_total, fail_num_total, 0, y, z, ip, version, env_type, '',presure])

    return data_list, data_detail_list


result_list = []

def get_list_moe(op_name):
    file_list=[]
    case_name_list=[]
    fail_case_name_list=[]
    if  os.path.exists("MC2_result_{}.csv".format(op_name)):
        os.remove("MC2_result_{}.csv".format(op_name))
    if not os.path.exists("MC2_result_{}.csv".format(op_name)):
        init_csv("MC2_result_{}.csv".format(op_name))
    if op_name=="moe":
        for file in os.listdir(os.getcwd()):
            if "moe" in file and "MC2" not in file:
                file_list.append(file)
    data = []
    for file in file_list:
        df = pandas.read_csv(file, names=['case_name', 'param', 'precision'])
        for case_name in df['case_name']:
            if case_name not in case_name_list:
                case_name_list.append(case_name)
        data_tmp = df.values.tolist()
        data.extend(data_tmp)

        try:

            for tep in data:
                if "FAIL" in tep[-1] and "ep_recv_counts" not in tep[1]:
                    if tep[0] not in fail_case_name_list:
                        fail_case_name_list.append(tep[0])
        except Exception as e:
            print(e)
    for case_name in case_name_list:
        if case_name  not in  fail_case_name_list:
            result_list.append([case_name,'mc2_test','', version,'success', formatted_time, formatted_time,'1','1','0'])
        else:
            result_list.append([case_name,'mc2_test','', version,'failed', formatted_time, formatted_time,'1','0','1'])
    if os.path.exists("case_name.csv"):
        allcase_list = readfile("case_name.csv")
        for case_name in allcase_list:
            if case_name not in case_name_list:
                result_list.append(
                    [case_name, 'mc2_test', '', version, 'failed', formatted_time, formatted_time, '1', '0', '1'])
    return result_list


def get_list_eplb(op_name):
    file_list=[]
    case_name_list=[]
    fail_case_name_list=[]
    if  os.path.exists("MC2_result_{}.csv".format(op_name)):
        os.remove("MC2_result_{}.csv".format(op_name))
    if not os.path.exists("MC2_result_{}.csv".format(op_name)):
        init_csv("MC2_result_{}.csv".format(op_name))
    if op_name=="eplb":
        for file in os.listdir(os.getcwd()):
            if "EPLB" in file and "MC2" not in file:
                file_list.append(file)
    data = []
    for file in file_list:
        df = pandas.read_csv(file, names=['case_name','param', 'param_1', 'precision'])
        for case_name in df['case_name']:
            if case_name not in case_name_list:
                case_name_list.append(case_name)
        data_tmp = df.values.tolist()
        data.extend(data_tmp)

        try:

            for tep in data:
                if "FAIL" in tep[-1] and "ep_recv_counts" not in tep[1]:
                    if tep[0] not in fail_case_name_list:
                        fail_case_name_list.append(tep[0])
        except Exception as e:
            print(e)
    for case_name in case_name_list:
        if case_name  not in  fail_case_name_list:
            result_list.append([case_name,'mc2_test','', version,'success', formatted_time, formatted_time,'1','1','0'])
        else:
            result_list.append([case_name,'mc2_test','', version,'failed', formatted_time, formatted_time,'1','0','1'])
    if os.path.exists("case_name.csv"):
        allcase_list = readfile("case_name.csv")
        for case_name in allcase_list:
            if case_name not in case_name_list:
                result_list.append(
                    [case_name, 'mc2_test', '', version, 'failed', formatted_time, formatted_time, '1', '0', '1'])
    return result_list

def get_list(op_name):
    result_case_list = []
    file_list=[]
    if  os.path.exists("MC2_result_{}.csv".format(op_name)):
        os.remove("MC2_result_{}.csv".format(op_name))
    if not os.path.exists("MC2_result_{}.csv".format(op_name)):
        init_csv("MC2_result_{}.csv".format(op_name))

    file_list=["{}_result.csv".format(op_name)]
    for precision_file in file_list:
        if precision_file.startswith("allgather") or precision_file.startswith("mm_allreduce_ARN_result")  or precision_file.startswith("aclnnAllGatherMatmul_result") or precision_file.startswith("alltoallv_gmm_result") or precision_file.startswith("gmm_alltoallv_result") :
            data = pandas.read_csv(precision_file, names=["case_name", "rank", "result", "gather"])
        else:
            data = pandas.read_csv(precision_file, names=["case_name", "rank", "result"])

        list_2 = data.values.tolist()
        case_name_list = set()
        for s in list_2:
            case_name_list.add(s[0])
        case_name_list = list(case_name_list)
        for case_name in case_name_list:
            pass_count = 0
            for num in list_2:
                if num[0] == case_name:
                    try:
                        if  "PASS" in num[2] or "Pass" in num[2]:
                            pass_count += 1
                    except Exception as  e:
                        print(e)

            if pass_count == 2 or pass_count >= 4 :
                precision = "PASS"
                result_list.append([case_name,'mc2_test','', version,'success', formatted_time, formatted_time,'1','1','0'])
            else:
                precision = "FAIL"
                result_list.append([case_name,'mc2_test','', version,'failed', formatted_time, formatted_time,'1','0','1'])
    if os.path.exists("case_name.csv"):
        allcase_list = readfile("case_name.csv")
        for case_name in allcase_list:
            if case_name not in case_name_list:
                result_list.append(
                    [case_name, 'mc2_test', '', version, 'failed', formatted_time, formatted_time, '1', '0', '1'])
    return result_list


def get_fuzz(op_name,loop,fail_num):
    result_case_list = []
    file_list=[]
    if  os.path.exists("MC2_result_{}.csv".format(op_name)):
        os.remove("MC2_result_{}.csv".format(op_name))
    if not os.path.exists("MC2_result_{}.csv".format(op_name)):
        init_csv("MC2_result_{}.csv".format(op_name))

    file_list=["{}_result.csv".format(op_name)]
    start=random.randint(1,10000)
    for i in range(start,start+int(loop)):
        case_name="{}_{}".format(op_name,i)
        result_list.append(
            [case_name, 'mc2_test', '', version, 'success', formatted_time, formatted_time, '1', '1', '0'])
    if int(fail_num) >0:
        for i in range(int(fail_num)):
            result_list[i][4]='failed'
            result_list[i][8]='0'
            result_list[i][9]='1'
    return result_list


if __name__ == '__main__':
    op_name=sys.argv[1]
    if op_name=="moe":
        get_list_moe(op_name)
    elif op_name=="eplb":
        get_list_eplb(op_name)
    elif "fuzz" in op_name:
        loop=sys.argv[2]
        fail_num=0
        if len(sys.argv) ==4:
            fail_num=sys.argv[3]
        print(fail_num)
        get_fuzz(op_name,loop,fail_num)
    else:
        get_list(op_name)
    for result in result_list:
            write_csv("MC2_result_{}.csv".format(op_name), result)
    ip=get_host_ip()
    report="MC2_result_{}.csv".format(op_name)
    taskid="mc2_test00{}".format(random.randint(1,10000))
    env_type = 1982
    if taskid:
        taskid_query = query_112(f"select * from op_alpha where 任务ID='{taskid}'")
        if taskid_query:
            print(f'{taskid}已经入过库，请更换后重试!')
        else:
            if report:
                network = report.split('/')[-1].replace('MC2_result_', '').replace('.csv', '')
                data_list, data_detail_list = get_mc2_data(taskid, network, report, ip, env_type)
                if data_list:
                    sql = "insert into op_alpha(组件,用例名,任务ID,状态,开始时间,结束时间,用例总数,执行成功数,执行失败数,已分析数,执行率,执行成功率,ips,version,env_type,invalid,pressure) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    insert_112(sql, data_list)
                if data_detail_list:
                    sql = "insert into op_alpha_detail(任务ID,用例名,结束时间,version,组件,env_type,用例结果) values (%s,%s,%s,%s,%s,%s,%s)"
                    insert_112(sql, data_detail_list)




