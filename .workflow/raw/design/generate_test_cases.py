#!/usr/bin/env python3
"""
GMM A16W4 PerGroup ND 测试用例生成脚本

用法: python generate_test_cases.py [--output PATH]
"""
import argparse, sys
try:
    import pandas as pd
except ImportError:
    print("Error: pip install pandas openpyxl"); sys.exit(1)



def generate_tiling_exception_cases():
    c = []
    # Helper for multi-multi base (groupType=-1)
    def mm(cid, desc, xd="FLOAT16", wd="INT4", bd="", sd="FLOAT16", od="", gd="",
           xs="256,512", ws="256,512", bs="", ss="4,256", os_="", gls="",
           gs=128, tw=0, glt="", glc="", exp="ERR", pri="P0", rem=""):
        c.append({"case_id":cid,"description":desc,"group_type":-1,"x_dtype":xd,"weight_dtype":wd,
                  "bias_dtype":bd,"antiquant_scale_dtype":sd,"antiquant_offset_dtype":od,
                  "group_list_dtype":gd,"x_shape":xs,"weight_shape":ws,"bias_shape":bs,
                  "antiquant_scale_shape":ss,"antiquant_offset_shape":os_,"group_list_shape":gls,
                  "group_size":gs,"transpose_weight":tw,"group_list_type":glt,
                  "group_list_content":glc,"expected_result":exp,"priority":pri,"remarks":rem})
    # Helper for single-single base (groupType=0)
    def ss(cid, desc, xd="FLOAT16", wd="INT4", bd="", sd="FLOAT16", od="", gd="INT64",
           xs="256,512", ws="4,256,512", bs="", ss_="4,4,256", os_="", gls="4",
           gs=128, tw=0, glt=0, glc="64,128,192,256", exp="ERR", pri="P0", rem=""):
        c.append({"case_id":cid,"description":desc,"group_type":0,"x_dtype":xd,"weight_dtype":wd,
                  "bias_dtype":bd,"antiquant_scale_dtype":sd,"antiquant_offset_dtype":od,
                  "group_list_dtype":gd,"x_shape":xs,"weight_shape":ws,"bias_shape":bs,
                  "antiquant_scale_shape":ss_,"antiquant_offset_shape":os_,"group_list_shape":gls,
                  "group_size":gs,"transpose_weight":tw,"group_list_type":glt,
                  "group_list_content":glc,"expected_result":exp,"priority":pri,"remarks":rem})

    # 1.1 dtype
    mm("T-01","x dtype非法(FLOAT32)",xd="FLOAT32",sd="FLOAT32",exp="ERR_INVALID_DTYPE",rem="x仅支持FLOAT16/BFLOAT16")
    mm("T-02","weight dtype非法(INT8)",wd="INT8",exp="ERR_INVALID_DTYPE",rem="weight仅支持INT4")
    mm("T-03","bias dtype与x不匹配: x=FP16,bias=BF16",bd="BFLOAT16",bs="256",exp="ERR_INVALID_DTYPE",rem="x=FP16时bias仅支持FLOAT16/null")
    mm("T-04","bias dtype与x不匹配: x=BF16,bias=FP16",xd="BFLOAT16",sd="BFLOAT16",bd="FLOAT16",bs="256",exp="ERR_INVALID_DTYPE",rem="x=BF16时bias仅支持BFLOAT16/FLOAT32/null")
    mm("T-05","antiquantScale dtype与x不一致: x=FP16,scale=BF16",sd="BFLOAT16",exp="ERR_INVALID_DTYPE",rem="antiquantScale dtype必须与x一致")
    mm("T-06","antiquantOffset dtype与x不一致: x=FP16,offset=BF16",od="BFLOAT16",os_="4,256",exp="ERR_INVALID_DTYPE",rem="antiquantOffset dtype必须与x一致")
    ss("T-07","groupList dtype非法(INT32)",gd="INT32",exp="ERR_INVALID_DTYPE",rem="groupList仅支持INT64")

    # 1.2 shape - 维度
    mm("T-08","x维度不为2(3D)",xs="4,64,512",exp="ERR_INVALID_DIM",rem="多多多x必须为2维")
    mm("T-09","多多多weight维度不为2(3D)",ws="4,256,512",exp="ERR_INVALID_DIM",rem="多多多weight必须为2维")
    ss("T-10","单单单weight维度不为3(2D)",ws="256,512",exp="ERR_INVALID_DIM",rem="单单单weight必须为3维")
    mm("T-11","多多多bias维度不为1(2D)",bd="FLOAT16",bs="4,256",exp="ERR_INVALID_DIM",pri="P1",rem="多多多bias必须为1维")
    ss("T-12","单单单bias维度不为2(1D)",bd="FLOAT16",bs="256",exp="ERR_INVALID_DIM",pri="P1",rem="单单单bias必须为2维")
    mm("T-13","多多多antiquantScale维度错误(3D)",ss="1,4,256",exp="ERR_INVALID_DIM",rem="多多多pergroup scale应为2维(G_i,N_i)")
    ss("T-14","单单单pergroup antiquantScale维度错误(2D)",ss_="4,256",exp="ERR_INVALID_DIM",rem="单单单pergroup scale应为3维(E,G,N)")
    ss("T-15","groupList维度不为1(2D)",gls="2,2",exp="ERR_INVALID_DIM",pri="P1",rem="groupList必须为1维")

    # 1.2 shape - 数值约束
    mm("T-16","K不是group_size整数倍: K=100,gs=32",xs="256,100",ws="256,100",ss="3,256",gs=32,exp="ERR_K_NOT_ALIGNED",rem="100%32=4!=0")
    mm("T-17","group_size不在合法集合: gs=48",ss="10,256",gs=48,exp="ERR_INVALID_GROUP_SIZE",rem="仅支持32/64/128/256")
    mm("T-18","weight最后一维(N)非偶数: N=3",ws="3,512",ss="4,3",exp="ERR_WEIGHT_LAST_DIM_ODD",rem="非转置时N必须为偶数")
    mm("T-19","pergroup数G_i不整除k_i: k=512,G=3",ss="3,256",exp="ERR_PERGROUP_NOT_DIVISIBLE",rem="512/3不整除")
    mm("T-20","多多多pergroup长度s_i不一致",xs="256,512",ws="256,256",exp="ERR_PERGROUP_LENGTH_INCONSISTENT",rem="tensor0:k=512,s=128; tensor1:k=256,s=64")
    mm("T-21","pergroup转置s_i非偶数: gs=33",xs="256,528",ws="528,256",ss="16,256",gs=33,tw=1,exp="ERR_PERGROUP_TRANSPOSE_ODD",pri="P1",rem="转置时s_i必须为偶数")

    # 1.2 shape - shape匹配
    mm("T-22","x.k与weight.k不匹配: xK=512,wK=256",ws="256,256",exp="ERR_SHAPE_MISMATCH",rem="非转置:weight K应与x K一致")
    mm("T-23","antiquantScale N与weight N不匹配",ss="4,128",exp="ERR_SCALE_SHAPE_MISMATCH",rem="scale N_i应与weight N一致")
    mm("T-24","antiquantScale G与K/gs不匹配: G=5,K/gs=4",ss="5,256",exp="ERR_SCALE_SHAPE_MISMATCH",rem="G_i应等于K/gs")
    mm("T-25","bias N与weight N不匹配",bd="FLOAT16",bs="128",exp="ERR_SHAPE_MISMATCH",pri="P1",rem="bias N_i应与weight N一致")
    ss("T-26","groupList长度与E不匹配: len=3,E=4",gls="3",glc="64,128,192",exp="ERR_SHAPE_MISMATCH",pri="P1",rem="groupList长度应等于E")

    # 1.3 format
    mm("T-27","x format非ND(NZ)",exp="ERR_INVALID_FORMAT",rem="x format=NZ,仅支持ND")
    mm("T-28","weight format非ND(NZ)",exp="ERR_INVALID_FORMAT",rem="weight format=NZ,仅支持ND")
    mm("T-29","bias format非ND",bd="FLOAT16",bs="256",exp="ERR_INVALID_FORMAT",pri="P1",rem="bias format=NZ,仅支持ND")
    mm("T-30","antiquantScale format非ND",exp="ERR_INVALID_FORMAT",pri="P1",rem="scale format=NZ,仅支持ND")
    mm("T-31","antiquantOffset format非ND",od="FLOAT16",os_="4,256",exp="ERR_INVALID_FORMAT",pri="P1",rem="offset format=NZ,仅支持ND")

    # 1.4 属性
    mm("T-32","groupType非法: 传入1",exp="ERR_INVALID_ATTR",rem="groupType仅支持-1/0")
    # Override group_type for T-32
    c[-1]["group_type"] = 1
    ss("T-33","groupListType非法: 传入2",glt=2,exp="ERR_INVALID_ATTR",rem="groupListType仅支持0/1")
    mm("T-34","actType非法: 传入1",exp="ERR_INVALID_ATTR",pri="P1",rem="actType仅支持0")
    mm("T-35","多多多splitItem非法: 传入2",exp="ERR_INVALID_ATTR",rem="多多多splitItem仅支持{0,1}")
    ss("T-36","单单单splitItem非法: 传入0",exp="ERR_INVALID_ATTR",rem="单单单splitItem仅支持{2,3}")
    mm("T-37","tuningConfigOptional传入非空",exp="ERR_INVALID_ATTR",pri="P1",rem="Ascend 950不支持此参数")

    # 1.5 场景校验
    ss("T-38","groupType=0但groupList为空",gd="",gls="",exp="ERR_GROUPLIST_REQUIRED",rem="groupType=0时groupList必传")
    mm("T-39","groupType=-1但groupList非空",gd="INT64",gls="4",glc="64,128,192,256",exp="ERR_GROUPLIST_NOT_ALLOWED",rem="groupType=-1时groupList必须传空")
    ss("T-40","groupListType=0数列非单调非递减",glc="64,32,192,256",exp="ERR_GROUPLIST_INVALID",pri="P1",rem="cumsum要求非负单调非递减")
    ss("T-41","groupListType=1数值为负",glt=1,glc="64,-32,128,96",exp="ERR_GROUPLIST_INVALID",pri="P1",rem="count要求非负")
    ss("T-42","groupListType=0最后值>x第一维: 300>256",glc="64,128,200,300",exp="ERR_GROUPLIST_INVALID",pri="P1",rem="cumsum最后值300>M=256")
    ss("T-43","groupListType=1总和>x第一维: 300>256",glt=1,glc="64,96,80,60",exp="ERR_GROUPLIST_INVALID",pri="P1",rem="count总和300>M=256")
    mm("T-44","多多多weight转置状态不统一",ws="256,512(mixed)",exp="ERR_TRANSPOSE_INCONSISTENT",rem="所有weight tensor转置状态须统一")

    return c


def generate_precision_cases():
    c = []
    def p(cid, desc, xd, bd, od, sd=None, exp="PASS", pri="P0", rem=""):
        if sd is None: sd = xd
        c.append({"case_id":cid,"description":desc,"group_type":-1,"x_dtype":xd,"weight_dtype":"INT4",
                  "bias_dtype":bd,"antiquant_scale_dtype":sd,"antiquant_offset_dtype":od,
                  "group_list_dtype":"","x_shape":"256,512","weight_shape":"256,512",
                  "bias_shape":"256" if bd else "","antiquant_scale_shape":"4,256",
                  "antiquant_offset_shape":"4,256" if od else "","group_list_shape":"",
                  "group_size":128,"transpose_weight":0,"group_list_type":"",
                  "group_list_content":"","expected_result":exp,"priority":pri,"remarks":rem})

    p("P-01","FP16有bias(FP16)有offset(FP16)","FLOAT16","FLOAT16","FLOAT16",exp="PASS (rtol=1e-3,atol=1e-3)",rem="y=x*(w+offset)*scale+bias")
    p("P-02","FP16有bias(FP16)无offset(对称)","FLOAT16","FLOAT16","",exp="PASS (rtol=1e-3,atol=1e-3)",rem="y=x*w*scale+bias")
    p("P-03","FP16无bias有offset(FP16)","FLOAT16","","FLOAT16",exp="PASS (rtol=1e-3,atol=1e-3)",rem="y=x*(w+offset)*scale")
    p("P-04","FP16无bias无offset(对称)","FLOAT16","","",exp="PASS (rtol=1e-3,atol=1e-3)",rem="y=x*w*scale")
    p("P-05","BF16有bias(BF16)有offset(BF16)","BFLOAT16","BFLOAT16","BFLOAT16",exp="PASS (rtol=1e-2,atol=1e-2)",rem="BF16完整公式")
    p("P-06","BF16有bias(BF16)无offset(对称)","BFLOAT16","BFLOAT16","",exp="PASS (rtol=1e-2,atol=1e-2)",rem="BF16对称+bias")
    p("P-07","BF16有bias(FP32)无offset","BFLOAT16","FLOAT32","",exp="PASS (rtol=1e-2,atol=1e-2)",pri="P1",rem="BF16+FP32 bias混合精度")
    p("P-08","BF16无bias有offset(BF16)","BFLOAT16","","BFLOAT16",exp="PASS (rtol=1e-2,atol=1e-2)",rem="BF16有offset无bias")
    p("P-09","BF16无bias无offset(对称)","BFLOAT16","","",exp="PASS (rtol=1e-2,atol=1e-2)",rem="BF16最简模式")
    return c


def generate_functional_cases():
    c = []
    def ssg(cid, desc, gs, tw=0, glt=0, glc="64,128,192,256", xs="256,512", ws=None, ss_=None,
            bd="", bs="", os_="", E=4, pri="P0", rem=""):
        if ws is None: ws = f"{E},256,512" if tw==0 else f"{E},512,256"
        G = 512 // gs if "512" in xs.split(",")[-1] else int(xs.split(",")[-1]) // gs
        if ss_ is None: ss_ = f"{E},{G},256"
        c.append({"case_id":cid,"description":desc,"group_type":0,"x_dtype":"FLOAT16","weight_dtype":"INT4",
                  "bias_dtype":bd,"antiquant_scale_dtype":"FLOAT16","antiquant_offset_dtype":"",
                  "group_list_dtype":"INT64","x_shape":xs,"weight_shape":ws,"bias_shape":bs,
                  "antiquant_scale_shape":ss_,"antiquant_offset_shape":os_,"group_list_shape":str(E),
                  "group_size":gs,"transpose_weight":tw,"group_list_type":glt,
                  "group_list_content":glc,"expected_result":"PASS","priority":pri,"remarks":rem})

    def mmg(cid, desc, gs, tw=0, E=4, ms="64", ks="512", ns="256", bd="", bs="", pri="P0", rem=""):
        G = int(ks) // gs
        ws = f"{ns},{ks}" if tw==0 else f"{ks},{ns}"
        ss_ = f"{G},{ns}"
        c.append({"case_id":cid,"description":desc,"group_type":-1,"x_dtype":"FLOAT16","weight_dtype":"INT4",
                  "bias_dtype":bd,"antiquant_scale_dtype":"FLOAT16","antiquant_offset_dtype":"",
                  "group_list_dtype":"","x_shape":f"{ms},{ks}","weight_shape":ws,"bias_shape":bs,
                  "antiquant_scale_shape":ss_,"antiquant_offset_shape":"","group_list_shape":"",
                  "group_size":gs,"transpose_weight":tw,"group_list_type":"",
                  "group_list_content":"","expected_result":"PASS","priority":pri,"remarks":rem})

    # 单单单
    ssg("F-01","单单单,非转置,gs=32",32,rem="G=512/32=16")
    ssg("F-02","单单单,非转置,gs=64",64,rem="G=512/64=8")
    ssg("F-03","单单单,非转置,gs=128",128,rem="G=512/128=4")
    ssg("F-04","单单单,非转置,gs=256",256,rem="G=512/256=2")
    ssg("F-05","单单单,转置,gs=128",128,tw=1,rem="转置:weight shape=(E,K,N)")
    ssg("F-06","单单单,groupListType=0(cumsum)",128,glt=0,rem="cumsum:[64,128,192,256]→m=[64,64,64,64]")
    ssg("F-07","单单单,groupListType=1(count)",128,glt=1,glc="64,64,64,64",rem="count:[64,64,64,64]")
    ssg("F-08","单单单,M=1(极小M)",128,xs="1,512",glt=1,glc="1,0,0,0",pri="P1",rem="极小M,仅第1个group有数据")
    ssg("F-09","单单单,K=gs=128(最小K)",128,xs="256,128",ws="4,256,128",ss_="4,1,256",rem="K=gs→G=1")
    ssg("F-10","单单单,N=2(极小N,偶数约束)",128,ws="4,2,512",ss_="4,4,2",pri="P1",rem="N=2最小偶数N")
    ssg("F-11","单单单,E=1(单expert)",128,E=1,ws="1,256,512",ss_="1,4,256",glc="256",pri="P1",rem="单expert退化")
    ssg("F-12","单单单,E=1024(最大expert)",128,E=1024,xs="1024,512",ws="1024,256,512",ss_="1024,4,256",
        glt=1,glc="1,1,...,1(1024个1)",pri="P2",rem="最大E,每group m=1")

    # 多多多
    mmg("F-13","多多多,非转置,gs=32",32,rem="G=512/32=16")
    mmg("F-14","多多多,非转置,gs=64",64,rem="G=512/64=8")
    mmg("F-15","多多多,非转置,gs=128",128,rem="G=512/128=4")
    mmg("F-16","多多多,非转置,gs=256",256,rem="G=512/256=2")
    mmg("F-17","多多多,转置,gs=128",128,tw=1,rem="转置:weight shape=(K,N)")
    mmg("F-18","多多多,m_i=1(极小M)",128,ms="1",pri="P1",rem="每个tensor m_i=1")
    mmg("F-19","多多多,k_i=gs=128(最小K)",128,ks="128",pri="P1",rem="K=gs→G=1")
    mmg("F-20","多多多,n_i=2(极小N)",128,ns="2",pri="P1",rem="N=2最小偶数N")
    mmg("F-21","多多多,E=2",128,E=2,ms="128",pri="P1",rem="2个expert")
    mmg("F-22","多多多,E=8",128,E=8,ms="32",rem="典型MoE 8experts")
    mmg("F-23","多多多,E=64",128,E=64,ms="4",pri="P1",rem="64个expert")
    mmg("F-24","多多多,E=128",128,E=128,ms="2",pri="P2",rem="128个expert")
    mmg("F-25","多多多,各expert m_i不同",128,pri="P1",rem="m=[1,32,64,128],k=512,n=256")
    c[-1]["x_shape"] = "varies:[1,512],[32,512],[64,512],[128,512]"

    # 完整参数组合
    ssg("F-26","单单单+bias(FP16)+offset(FP16),非转置,gs=128",128,
        bd="FLOAT16",bs="4,256",os_="4,4,256",rem="完整参数:bias+offset+单单单")
    mmg("F-27","多多多+bias(FP16)+offset(FP16),转置,gs=128",128,tw=1,
        bd="FLOAT16",bs="256",rem="完整参数:bias+offset+多多多+转置")
    c[-1]["antiquant_offset_dtype"] = "FLOAT16"
    c[-1]["antiquant_offset_shape"] = "4,256"
    c[-2]["antiquant_offset_dtype"] = "FLOAT16"

    return c


def generate_performance_cases():
    c = []
    def perf(cid, desc, gt, E, M, K, N, gs, tw=0, gd="INT64" if 0 else "", glt="", glc="", pri="P0", rem=""):
        G = K // gs
        if gt == -1:
            ws = f"{N},{K}" if tw==0 else f"{K},{N}"
            ss_ = f"{G},{N}"
            gls = ""
        else:
            ws = f"{E},{N},{K}" if tw==0 else f"{E},{K},{N}"
            ss_ = f"{E},{G},{N}"
            gls = str(E)
        c.append({"case_id":cid,"description":desc,"group_type":gt,"x_dtype":"FLOAT16","weight_dtype":"INT4",
                  "bias_dtype":"","antiquant_scale_dtype":"FLOAT16","antiquant_offset_dtype":"",
                  "group_list_dtype":gd if gt==0 else "","x_shape":f"{M},{K}","weight_shape":ws,"bias_shape":"",
                  "antiquant_scale_shape":ss_,"antiquant_offset_shape":"","group_list_shape":gls,
                  "group_size":gs,"transpose_weight":tw,"group_list_type":glt if gt==0 else "",
                  "group_list_content":glc,"expected_result":"PASS+性能基线","priority":pri,"remarks":rem})

    perf("PERF-01","典型MoE: E=8,M=256,K=4096,N=4096,gs=128",-1,8,256,4096,4096,128,rem="典型MoE gate/up; G=32")
    perf("PERF-02","典型MoE: E=64,M=32,K=4096,N=4096,gs=128",-1,64,32,4096,4096,128,rem="大规模MoE; G=32")
    perf("PERF-03","大shape: E=1,M=1024,K=8192,N=8192,gs=128",-1,1,1024,8192,8192,128,pri="P1",rem="验证tiling切分; G=64")
    perf("PERF-04","小shape: E=1,M=1,K=128,N=128,gs=128",-1,1,1,128,128,128,pri="P1",rem="验证尾块处理")
    perf("PERF-05","典型MoE+转置: E=8,M=256,K=4096,N=4096,gs=128",-1,8,256,4096,4096,128,tw=1,pri="P1",rem="转置性能")
    perf("PERF-06","单单单大E: E=256,M=256,K=4096,N=1024,gs=128",0,256,256,4096,1024,128,
         glt=1,glc="1,1,...,1(256个1)",pri="P1",rem="单单单大E; G=32")
    return c


def generate_generalization_cases():
    c = []
    def gen(cid, desc, gt, xs, ws, ss_, gs, tw=0, gd="", glt="", glc="", pri="P1", rem=""):
        c.append({"case_id":cid,"description":desc,"group_type":gt,"x_dtype":"FLOAT16","weight_dtype":"INT4",
                  "bias_dtype":"","antiquant_scale_dtype":"FLOAT16","antiquant_offset_dtype":"",
                  "group_list_dtype":gd,"x_shape":xs,"weight_shape":ws,"bias_shape":"",
                  "antiquant_scale_shape":ss_,"antiquant_offset_shape":"","group_list_shape":str(len(glc.split(","))) if glt else "",
                  "group_size":gs,"transpose_weight":tw,"group_list_type":glt,
                  "group_list_content":glc,"expected_result":"PASS","priority":pri,"remarks":rem})

    gen("G-01","gs=32(最小),多多多",-1,"64,256","128,256","8,128",32,rem="最小gs; G=256/32=8")
    gen("G-02","gs=256(最大),多多多",-1,"64,1024","256,1024","4,256",256,rem="最大gs; G=1024/256=4")
    gen("G-03","K=gs=128(单group),多多多",-1,"64,128","256,128","1,256",128,rem="K=gs→G=1,退化perchannel")
    gen("G-04","K=gs*10=1280(多group),多多多",-1,"64,1280","256,1280","10,256",128,rem="G=10,多group遍历")
    gen("G-05","M=1(极小M),多多多",-1,"1,512","256,512","4,256",128,rem="极小M边界")
    gen("G-06","N=2(极小N,偶数约束),多多多",-1,"64,512","2,512","4,2",128,rem="N=2最小偶数N")
    gen("G-07","E=1(单expert),多多多",-1,"256,512","256,512","4,256",128,rem="多多多退化为单matmul")
    gen("G-08","E=1024(最大expert),单单单",0,"1024,512","1024,256,512","1024,4,256",128,
        gd="INT64",glt=1,glc="1,1,...,1(1024个1)",pri="P2",rem="groupList最大长度1024")
    gen("G-09","各expert m_i极端不同,多多多",-1,"varies:[1,512],[16,512],[64,512],[256,512]",
        "256,512(each)","4,256(each)",128,rem="极端不均匀m_i")
    gen("G-10","各expert k_i不同(gs相同),多多多",-1,"varies:[64,256],[64,512]",
        "varies:[128,256],[256,512]","varies:[2,128],[4,256]",128,rem="k_i不同但s_i=128一致")
    gen("G-11","单单单cumsum非均匀分组",0,"256,512","4,256,512","4,4,256",128,
        gd="INT64",glt=0,glc="1,33,97,256",rem="cumsum非均匀:m=[1,32,64,159]")
    gen("G-12","单单单count非均匀分组",0,"256,512","4,256,512","4,4,256",128,
        gd="INT64",glt=1,glc="1,32,64,159",rem="count非均匀:sum=256=M")
    return c


def generate_kernel_stub_cases():
    c = []
    def ks(cid, desc, stage, loc, inp, xs, ws, ss, os_, gs, tw, gt, exp, pri="P0", rem=""):
        c.append({"case_id":cid,"description":desc,"verification_stage":stage,"stub_location":loc,
                  "input_data_pattern":inp,"x_shape":xs,"weight_shape":ws,
                  "antiquant_scale_shape":ss,"antiquant_offset_shape":os_,
                  "group_size":gs,"transpose_weight":tw,"group_type":gt,
                  "expected_output":exp,"priority":pri,"remarks":rem})

    # MTE2
    ks("KS-01","Weight INT4 GM→UB搬运","MTE2","CopyWeightGmToUb后,输出UB到GM",
       "等差数列(0,1,...,15重复)","16,128","128,128","1,128","",128,0,-1,
       "UB中INT4数据与GM逐字节一致",rem="验证MTE2搬运INT4原始数据")
    ks("KS-02","Scale GM→UB搬运(pergroup 2D)","MTE2","CopyAntiQuantParamsGmToUb后,输出UB scale到GM",
       "scale[g,n]=g*100+n","16,256","256,256","2,256","",128,0,-1,
       "UB中scale布局正确:shape=(2,256),值与GM一致",rem="gs=128,kL1=256→scale=(2,256)")
    ks("KS-03","Offset GM→UB搬运(pergroup 2D)","MTE2","CopyAntiQuantParamsGmToUb后,输出UB offset到GM",
       "offset[g,n]=g*10+n","16,256","256,256","2,256","2,256",128,0,-1,
       "UB中offset布局正确,值与GM一致",rem="验证offset 2D搬运")

    # MTE3
    ks("KS-04","反量化后weight UB→L1搬运","MTE3","WeightHighBitUbToL1后,输出L1到GM",
       "全1.0 FP16","16,128","128,128","1,128","",128,0,-1,
       "L1中数据与UB一致(含ND→NZ排布转换)",rem="验证MTE3+排布转换")

    # MTE1
    ks("KS-05","A矩阵GM→L1搬运","MTE1","CopyAAndBiasGmToL1后,输出AL1到GM",
       "等差数列A[m,k]=m*K+k","16,128","128,128","1,128","",128,0,-1,
       "AL1中数据与GM中A一致",rem="验证MTE1 A矩阵搬运")
    ks("KS-06","Bias GM→L1搬运","MTE1","CopyAAndBiasGmToL1后,输出BiasL1到GM",
       "bias[n]=n","16,128","128,128","1,128","",128,0,-1,
       "BiasL1中数据与GM一致",pri="P1",rem="验证bias搬运到L1")

    # 输出写回
    ks("KS-07","C矩阵L1→GM写回","MTE1","GetTensorC后,检查GM输出",
       "L1中C全2.0","16,128","128,128","1,128","",128,0,-1,
       "GM输出与L1中C数据一致",rem="验证C写回GM")

    # VF - INT4 unpack/cast
    ks("KS-08","INT4 unpack正确性(UNPACK4)","VF","DataCopy UNPACK4后,输出unpack结果到GM",
       "INT4:0,1,...,7,-8,...,-1(完整范围)","16,128","128,128","1,128","",128,0,-1,
       "unpack后INT8值与预期一致(每字节拆为2个INT4)",rem="验证UNPACK4指令")
    ks("KS-09","INT4→FP16 cast正确性","VF","Cast后,输出FP16结果到GM",
       "unpack后INT8数据","16,128","128,128","1,128","",128,0,-1,
       "FP16值与INT8数值相等",rem="验证Cast int4→fp16")

    # VF - scale/offset加载
    ks("KS-10","NK排布scale广播加载","VF","NdNkLoadScaleOffset后,输出scale VREG到GM",
       "scale[g,n]=g*100+n","16,256","256,256","2,256","",128,0,-1,
       "每个weight元素对应scale值正确(broadcast)",rem="gs=128,nOffset=0,kOffset=0")
    ks("KS-11","KN排布scale直接加载","VF","NdKnLoadScaleOffset后,输出scale VREG到GM",
       "scale[g,n]=g*100+n","16,256","256,256","2,256","",128,0,-1,
       "每个weight元素对应scale值正确(direct)",rem="KN排布DIST_NORM加载")
    ks("KS-12","NK排布offset广播加载","VF","NdNkLoadScaleOffset后,输出offset VREG到GM",
       "offset[g,n]=g*10+n","16,256","256,256","2,256","2,256",128,0,-1,
       "每个weight元素对应offset值正确",pri="P1",rem="验证offset broadcast")

    # VF - 反量化计算
    ks("KS-13","反量化公式(有offset)","VF","Add+Mul后,输出结果到GM",
       "w=2,scale=1.5,offset=1","16,128","128,128","1,128","1,128",128,0,-1,
       "(2+1)*1.5=4.5(容差1e-3)",rem="验证(w+offset)*scale")
    ks("KS-14","反量化公式(无offset,对称)","VF","Mul后,输出结果到GM",
       "w=2,scale=1.5","16,128","128,128","1,128","",128,0,-1,
       "2*1.5=3.0(容差1e-3)",rem="验证w*scale")

    # 地址偏移
    ks("KS-15","PerGroup scale偏移(单单单)","Address","SetGmAddr后,输出antiquantParamsBaseOffset到GM",
       "E=4,K=512,N=256,gs=128","256,512","4,256,512","4,4,256","",128,0,0,
       "group偏移:0,1024,2048,3072 (每group=(K/gs)*N=4*256=1024)",rem="验证pergroup scale偏移公式")
    ks("KS-16","Bias偏移分离(单单单)","Address","SetGmAddr后,输出biasBaseOffset到GM",
       "E=4,N=256","256,512","4,256,512","4,4,256","",128,0,0,
       "bias偏移:0,256,512,768 (每group=N=256)",rem="验证bias独立偏移")
    ks("KS-17","INT4 weight偏移>>1(单单单)","Address","SetGmAddr后,输出weightBaseOffset到GM",
       "E=4,K=512,N=256","256,512","4,256,512","4,4,256","",128,0,0,
       "weight偏移:0,65536,131072,196608 (每group=(N*K)>>1=65536)",rem="验证INT4偏移>>1")

    # Tiling参数传递
    ks("KS-18","groupSize传递","Tiling","Controller Init后,输出groupSize到GM",
       "gs=128","16,512","256,512","4,256","",128,0,-1,
       "kernel中groupSize=128,与tiling一致",rem="验证tiling→kernel参数传递")
    ks("KS-19","mList/kList/nList(多多多)","Tiling","Process循环中,输出list到GM",
       "E=4,m=[1,32,64,128],k=512,n=256","varies","256,512(each)","4,256(each)","",128,0,-1,
       "list值与tiling设置一致",rem="验证多多多list传递")
    ks("KS-20","baseM/baseN动态调整","Tiling","Process中,输出调整后baseM/baseN到GM",
       "M=100,N=200","100,512","256,200","4,200","",128,0,-1,
       "baseM=50(M<256→CeilDiv(100,2)),baseN=128(256>>1)",pri="P1",rem="验证动态调整逻辑")

    # Matmul
    ks("KS-21","Matmul基础计算","Matmul","LaunchMatmul后,输出C到GM",
       "A全1,W_dequant全1,bias全0","16,64","64,16","1,16","",64,0,-1,
       "C[m,n]=K=64(累加64次1*1)",rem="M=16,K=64,N=16")
    ks("KS-22","Matmul+bias","Matmul","LaunchMatmul后,输出C到GM",
       "A全1,W_dequant全1,bias[n]=n","16,64","64,16","1,16","",64,0,-1,
       "C[m,n]=K+n=64+n",rem="验证matmul+bias融合")

    return c


def generate_tiling_whitebox_cases():
    """生成 tiling 白盒用例 (TW-01 ~ TW-20)"""
    c = []
    def tw(cid, desc, code_path, gt, xd="FLOAT16", wd="INT4", bd="", sd="FLOAT16", od="",
            xs="", ws="", ss="", os_="", gs=128, tw_=0, glt="", exp_fields="", pri="P0", rem=""):
        c.append({"case_id": cid, "description": desc, "code_path": code_path,
                  "group_type": gt, "x_dtype": xd, "weight_dtype": wd,
                  "bias_dtype": bd, "antiquant_scale_dtype": sd, "antiquant_offset_dtype": od,
                  "x_shape": xs, "weight_shape": ws,
                  "antiquant_scale_shape": ss, "antiquant_offset_shape": os_,
                  "group_size": gs, "transpose_weight": tw_, "group_list_type": glt,
                  "expected_tiling_fields": exp_fields, "priority": pri, "remarks": rem})

    # 8.1 IsA16W4NDPergroup 判断路径
    tw("TW-01", "pergroup场景正确识别",
       "IsA16W4NDPergroup() → groupSize_>0 为true",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="groupSize=128 (K/G=512/4=128)", rem="单单单,E=4,M=256,K=512,N=256")
    tw("TW-02", "perchannel场景不误判为pergroup",
       "IsA16W4NDPergroup() → groupSize_==0 为true",
       0, xs="256,512", ws="4,256,512", ss="4,256", gs=0,
       exp_fields="groupSize=0", pri="P0", rem="单单单,perchannel,scale_shape=(E,N)=(4,256)")
    tw("TW-03", "多多多pergroup 2维scale正确识别",
       "SetAntiquantGroupSize()中antiquantScaleDimNum==2&&IsA16W4ND()分支",
       -1, xs="64,512", ws="256,512", ss="4,256", gs=128,
       exp_fields="groupSize=128 (K/G=512/4=128)", rem="多多多,E=4,scale_shape=(G_i,N_i)=(4,256)")

    # 8.2 SetAntiquantGroupSize 路径
    tw("TW-04", "单单单3维scale groupSize推导",
       "SetAntiquantGroupSize()中antiquantScaleDimNum==3分支",
       0, xs="256,1024", ws="4,256,1024", ss="4,8,256", gs=128,
       exp_fields="groupSize=128 (K/G=1024/8=128)", rem="单单单,E=4,K=1024,G=8")
    tw("TW-05", "group_size=32场景",
       "CheckGroupSize()中groupSize_==32分支",
       0, xs="256,256", ws="4,256,256", ss="4,8,256", gs=32,
       exp_fields="groupSize=32,校验通过", rem="单单单,K=256,G=8→gs=32")
    tw("TW-06", "group_size=64场景",
       "CheckGroupSize()中groupSize_==64分支",
       0, xs="256,256", ws="4,256,256", ss="4,4,256", gs=64,
       exp_fields="groupSize=64", rem="单单单,K=256,G=4→gs=64")
    tw("TW-07", "group_size=256场景",
       "CheckGroupSize()中groupSize_==256分支",
       0, xs="256,256", ws="4,256,256", ss="4,1,256", gs=256,
       exp_fields="groupSize=256", pri="P1", rem="单单单,K=256,G=1→gs=256")

    # 8.3 SetTilingKey 路径
    tw("TW-08", "pergroup antiquantType设置",
       "SetTilingKey()中IsA16W4NDPergroup()→antiquantType=PER_GROUP",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="W_QUANT_TYPE=WQGMM_PER_GROUP", rem="pergroup场景")
    tw("TW-09", "perchannel antiquantType设置(对比)",
       "SetTilingKey()中else分支→antiquantType=PER_CHANNEL",
       0, xs="256,512", ws="4,256,512", ss="4,256", gs=0,
       exp_fields="W_QUANT_TYPE=WQGMM_PER_CHANNEL", pri="P1", rem="perchannel场景对比")
    tw("TW-10", "转置场景tiling key",
       "SetTilingKey()中transB_==true→WQ_B_TRANS=WQGMM_TRANS",
       0, xs="256,512", ws="4,512,256", ss="4,4,256", gs=128, tw_=1,
       exp_fields="WQ_B_TRANS=1", rem="pergroup+转置")
    tw("TW-11", "非转置场景tiling key",
       "SetTilingKey()中transB_==false→WQ_B_TRANS=WQGMM_NO_TRANS",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128, tw_=0,
       exp_fields="WQ_B_TRANS=0", rem="pergroup+非转置")

    # 8.4 SetShapeList 路径
    tw("TW-12", "单单单shape list填充",
       "SetShapeListSplitMSingleXSingleWeightSingleY()",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="groupNum=4,mSize=256,kSize=512,nSize=256,kList[0]=512,nList[0]=256,mList[0]=-1",
       rem="单单单,E=4")
    tw("TW-13", "多多多shape list填充",
       "SetShapeListMultiXMultiWeightMultiY()",
       -1, xs="varies:[1,512],[32,512],[64,512],[128,512]", ws="256,512(each)", ss="4,256(each)", gs=128,
       exp_fields="mList[i]/kList[i]/nList[i]逐tensor正确填充", rem="多多多,E=4,m=[1,32,64,128]")
    tw("TW-14", "INT4 weight nSize扩展",
       "SetShapeListSplitMSingleXSingleWeightSingleY()中weightDtype_==DT_INT4&&!transB_→nSize_=8*nSize_",
       0, xs="256,512", ws="4,32,512", ss="4,4,256", gs=128,
       exp_fields="nSize=256 (32*8=256)", rem="INT4存储,N实际=32*8=256")

    # 8.5 baseM/baseN/baseK 计算路径
    tw("TW-15", "默认base值(无bias)",
       "SetBaseTiling()中默认值设置",
       0, xs="512,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="baseM=256,baseN=256,baseK=64", rem="pergroup,无bias,M=512,N=512")
    tw("TW-16", "有bias时baseM调整",
       "SetBaseTiling()中hasBias分支→baseM=240",
       0, bd="FLOAT16", xs="512,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="baseM=240", rem="pergroup,有bias,M=512,N=512")

    # 8.6 CalcResplitTiling 路径
    tw("TW-17", "多多多ND不使能尾块重切分",
       "EnableTailResplit()→多多多→不使能",
       -1, xs="64,512", ws="256,512", ss="4,256", gs=128,
       exp_fields="mainBlockSize=0(不使能标志)", pri="P1", rem="多多多+ND")
    tw("TW-18", "单单单ND非转置不使能尾块重切分",
       "EnableTailResplit()→单单单+ND+B非转置→不使能",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp_fields="mainBlockSize=0", pri="P1", rem="单单单+ND+非转置")

    # 8.7 GMMWeightQuantParam 字段完整性
    tw("TW-19", "完整参数传递验证(单单单)",
       "SetTiling()完整流程",
       0, bd="FLOAT16", xs="256,512", ws="4,256,512", ss="4,4,256", gs=128, glt=0,
       exp_fields="groupNum=4,coreNum>0,kSize=512,nSize=256,singleX=1,singleWeight=1,singleY=1,groupType=0,groupListType=0,hasBias=1,groupSize=128",
       rem="单单单,E=4,有bias,非转置")
    tw("TW-20", "完整参数传递验证(多多多)",
       "SetTiling()完整流程",
       -1, xs="varies:[1,512],[32,512],[64,512],[128,512]", ws="256,512(each)", ss="4,256(each)", gs=128,
       exp_fields="singleX=0,singleWeight=0,singleY=0,groupType=-1,hasBias=0,groupSize=128",
       rem="多多多,E=4,m=[1,32,64,128],无bias,非转置")

    return c


def generate_kernel_whitebox_cases():
    """生成 kernel 白盒用例 (KW-01 ~ KW-22)"""
    c = []
    def kw(cid, desc, code_path, gt, xd="FLOAT16", wd="INT4", bd="", sd="FLOAT16", od="",
            xs="", ws="", ss="", os_="", gs=128, tw_=0, exp="PASS",
            rtol="1e-3", atol="1e-3", pri="P0", rem=""):
        c.append({"case_id": cid, "description": desc, "code_path": code_path,
                  "group_type": gt, "x_dtype": xd, "weight_dtype": wd,
                  "bias_dtype": bd, "antiquant_scale_dtype": sd, "antiquant_offset_dtype": od,
                  "x_shape": xs, "weight_shape": ws,
                  "antiquant_scale_shape": ss, "antiquant_offset_shape": os_,
                  "group_size": gs, "transpose_weight": tw_,
                  "expected_result": exp, "rtol": rtol, "atol": atol,
                  "priority": pri, "remarks": rem})

    # 9.1 if constexpr 分支覆盖
    kw("KW-01", "pergroup无offset无bias非转置分支",
       "V310_GMM_ANTI_QUANT→W_TYPE==WQGMM_ND&&OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_NOT_EXIST&&W_QUANT_TYPE==WQGMM_PER_GROUP&&WQ_B_TRANS==WQGMM_NO_TRANS",
       -1, xs="256,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS (使用A16W4_NDKN_PG配置)", rem="多多多,pergroup,无offset,无bias,非转置")
    kw("KW-02", "pergroup无offset有bias非转置分支",
       "OFFSET_OR_BIAS_EXIT==NOT_EXIST_BIAS_EXIST",
       -1, bd="FLOAT16", xs="256,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS", rem="多多多,pergroup,无offset,有bias(FP16),非转置")
    kw("KW-03", "pergroup有offset无bias非转置分支",
       "OFFSET_OR_BIAS_EXIT==EXIST_BIAS_NOT_EXIST",
       -1, od="FLOAT16", xs="256,512", ws="256,512", ss="4,256", os_="4,256", gs=128,
       exp="PASS", rem="多多多,pergroup,有offset(FP16),无bias,非转置")
    kw("KW-04", "pergroup有offset有bias非转置分支",
       "OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST",
       -1, bd="FLOAT16", od="FLOAT16", xs="256,512", ws="256,512", ss="4,256", os_="4,256", gs=128,
       exp="PASS", rem="多多多,pergroup,有offset(FP16),有bias(FP16),非转置")
    kw("KW-05", "pergroup无offset无bias转置分支",
       "WQ_B_TRANS==WQGMM_TRANS→使用A16W4_NDNK_PG配置",
       -1, xs="256,512", ws="512,256", ss="4,256", gs=128, tw_=1,
       exp="PASS", rem="多多多,pergroup,无offset,无bias,转置")
    kw("KW-06", "pergroup有offset有bias转置分支",
       "WQ_B_TRANS==WQGMM_TRANS&&OFFSET_OR_BIAS_EXIT==EXIST_BIAS_EXIST",
       -1, bd="FLOAT16", od="FLOAT16", xs="256,512", ws="512,256", ss="4,256", os_="4,256", gs=128, tw_=1,
       exp="PASS", rem="多多多,pergroup,有offset(FP16),有bias(FP16),转置")

    # 9.2 ComputeBasicBlock 分发路径
    kw("KW-07", "ND非转置→NdKn路径",
       "ComputeBasicBlock()→(ND&&!bTrans)→ComputeBasicBlockAivNdKnNzNk()",
       -1, xs="256,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS (K为内轴路径)", rem="多多多,pergroup,ND,非转置")
    kw("KW-08", "ND转置→NdNk路径",
       "ComputeBasicBlock()→else→ComputeBasicBlockAivNdNkNzKn()",
       -1, xs="256,512", ws="512,256", ss="4,256", gs=128, tw_=1,
       exp="PASS (N为内轴路径)", rem="多多多,pergroup,ND,转置")

    # 9.3 GetBufferConfig 分支覆盖
    kw("KW-09", "pergroup ND buffer配置",
       "GetBufferConfig()→weightFormat==ND&&antiQuantType==PER_GROUP→GetNdPerGroupBufferInfo()",
       -1, xs="256,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS (scale UB=8KB配置被使用)", rem="多多多,pergroup,ND")
    kw("KW-10", "perchannel ND buffer配置(对比)",
       "GetBufferConfig()→weightFormat==ND&&antiQuantType!=PER_GROUP→GetNdBufferInfo()",
       -1, xs="256,512", ws="256,512", ss="1,256", gs=0,
       exp="PASS (原有perchannel路径不受影响)", pri="P1", rem="多多多,perchannel,ND,对比验证")

    # 9.4 VF 函数 groupSize 运行时分支覆盖
    kw("KW-11", "groupSize=32 VF分支",
       "AntiQuantInt4NdNkPerGroup()→groupSize==32分支",
       -1, xs="64,256", ws="256,256", ss="8,256", gs=32,
       exp="PASS", rem="多多多,pergroup,gs=32,K=256")
    kw("KW-12", "groupSize=64 VF分支",
       "AntiQuantInt4NdNkPerGroup()→groupSize==64分支",
       -1, xs="64,256", ws="256,256", ss="4,256", gs=64,
       exp="PASS", rem="多多多,pergroup,gs=64,K=256")
    kw("KW-13", "groupSize=128 VF分支",
       "AntiQuantInt4NdNkPerGroup()→groupSize==128分支",
       -1, xs="64,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS", rem="多多多,pergroup,gs=128,K=512")
    kw("KW-14", "groupSize=256 VF分支",
       "AntiQuantInt4NdNkPerGroup()→groupSize==256分支",
       -1, xs="64,512", ws="256,512", ss="2,256", gs=256,
       exp="PASS", rem="多多多,pergroup,gs=256,K=512")

    # 9.5 CopyAntiQuantParamsGmToUb PER_GROUP 分支
    kw("KW-15", "pergroup scale 2D copy(多多多)",
       "CopyAntiQuantParamsGmToUb()→PER_GROUP分支→2D copy",
       -1, xs="64,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS (scale正确加载到UB)", rem="多多多,pergroup,scale_shape=(G_i,N_i)=(4,256)")
    kw("KW-16", "pergroup scale 2D copy(单单单)",
       "CopyAntiQuantParamsGmToUb()→PER_GROUP分支→2D copy(单单单偏移)",
       0, xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp="PASS", rem="单单单,pergroup,scale_shape=(E,G,N)=(4,4,256)")

    # 9.6 SetGmAddr pergroup 偏移路径
    kw("KW-17", "pergroup单单单多group偏移序列",
       "SetGmAddr()→if constexpr(PER_GROUP)→antiquantParamsBaseOffset+=(kSize/gs)*nSize",
       0, xs="256,512", ws="8,256,512", ss="8,4,256", gs=128,
       exp="PASS (8个group的scale偏移序列正确)", rem="单单单,E=8,K=512,N=256,gs=128")
    kw("KW-18", "pergroup bias偏移分离",
       "SetGmAddr()→biasBaseOffset+=nSize(独立于antiquantParamsBaseOffset)",
       0, bd="FLOAT16", xs="256,512", ws="4,256,512", ss="4,4,256", gs=128,
       exp="PASS (bias和scale偏移独立)", rem="单单单,E=4,K=512,N=256,gs=128,有bias")

    # 9.7 AIC/AIV 协作路径
    kw("KW-19", "多BasicBlock迭代(M>baseM)",
       "Process()中baseM动态调整→多个BasicBlock迭代",
       -1, xs="1024,512", ws="256,512", ss="4,256", gs=128,
       exp="PASS (多块拼接正确)", rem="多多多,pergroup,M=1024,N=256,K=512")
    kw("KW-20", "多BasicBlock迭代(N>baseN)",
       "Process()中baseN动态调整→N方向多块",
       -1, xs="256,1024", ws="1024,512", ss="4,1024", gs=128,
       exp="PASS", rem="多多多,pergroup,M=256,N=1024,K=512")
    kw("KW-21", "K方向多kBlock迭代",
       "WeightAntiQuantCompute()中K方向多kBlock循环",
       -1, xs="256,2048", ws="256,2048", ss="16,256", gs=128,
       exp="PASS (K方向累加正确)", rem="多多多,pergroup,M=256,N=256,K=2048")
    kw("KW-22", "单BasicBlock(M≤baseM,N≤baseN)",
       "Process()中单块直接计算,无迭代",
       -1, xs="64,128", ws="64,128", ss="1,64", gs=128,
       exp="PASS", pri="P1", rem="多多多,pergroup,M=64,N=64,K=128")

    return c


def main():
    parser = argparse.ArgumentParser(description="GMM A16W4 PerGroup ND 测试用例生成")
    parser.add_argument("--output", "-o", default="test_cases_gmm_a16w4_pergroup_nd.xlsx",
                        help="输出Excel文件路径")
    args = parser.parse_args()

    sheets = [
        ("tiling_exception", generate_tiling_exception_cases()),
        ("precision", generate_precision_cases()),
        ("functional", generate_functional_cases()),
        ("performance", generate_performance_cases()),
        ("generalization", generate_generalization_cases()),
        ("kernel_stub", generate_kernel_stub_cases()),
        ("tiling_whitebox", generate_tiling_whitebox_cases()),
        ("kernel_whitebox", generate_kernel_whitebox_cases()),
    ]

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        for name, cases in sheets:
            df = pd.DataFrame(cases)
            df.to_excel(writer, sheet_name=name, index=False)
            print(f"  {name}: {len(cases)} cases")

    total = sum(len(cases) for _, cases in sheets)
    print(f"\nGenerated {args.output} ({total} total cases, {len(sheets)} sheets)")


if __name__ == "__main__":
    main()
