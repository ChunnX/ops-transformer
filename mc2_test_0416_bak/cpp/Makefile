#
#loading path
#--------------------------------------------------------------------------------------------------------------------------------------------------
$(info "SOC:" $(SOC_VERSON) )

CXXFLAGS := -std=c++11\
        -Werror\
        -fstack-protector-strong\
        -fPIE -pie\
        -O2\
        -s\
        -Wl,-z,relro\
        -Wl,-z,now\
        -Wl,-z,noexecstack\
        -Wl,--copy-dt-needed-entries

ifeq ($(SOC_VERSON),910D_MODEL)
CXXFLAGS += -DDAVID_FAST_MODEL
endif 

ifeq ($(SOC_VERSON),910D)
CXXFLAGS += -DDAVID
CXXFLAGS += -DDAVID_FAST_MODEL
endif 

ifeq ($(SOC_VERSON),910C)
CXXFLAGS += -DSOC_910C
endif
ifeq ($(SOC_VERSON),910B)
CXXFLAGS += -DSOC_910B
endif 

Common_SRC = $(wildcard ${Common_DIR}/*.cc)
Opbase_SRC = $(wildcard ${Opbase_DIR}/*.cc)

HCCL_INC_DIR = ${ASCEND_DIR}/include
HCCL_LIB_DIR = ${ASCEND_DIR}/lib64

ACL_INC_DIR = ${ASCEND_DIR}/
ACL_LIB_DIR = ${ASCEND_DIR}/aarch64-linux/lib64
CUSTOM_OPP_LIB_DIR = ${ASCEND_DIR}/opp/vendors/customize/op_api/lib/

MPI_INC_DIR = ${MPI_HOME}/include
MPI_LIB_DIR = ${MPI_HOME}/lib

#
#library flags
#--------------------------------------------------------------------------------------------------------------------------------------------------
LIBS = -L$(HCCL_LIB_DIR) -lhccl\
		-L$(ACL_LIB_DIR) -lascendcl\
    	-L$(ACL_LIB_DIR) -lruntime -lacl_cblas -lascendcl -lc_sec -lopapi -lnnopbase -lascendalog -lerror_manager -lge_common_base  \
		# -L$(MPI_LIB_DIR) -lmpi

CUSTOM_LIBS = -L$(HCCL_LIB_DIR) -lhccl\
		-L$(ACL_LIB_DIR) -lascendcl\
    	-L$(ACL_LIB_DIR) -lruntime -lacl_cblas -lascendcl -lc_sec -lopapi -lnnopbase -lascendalog -lerror_manager -lge_common_base  \
		# -L$(MPI_LIB_DIR) -lmpi \
		-L$(CUSTOM_OPP_LIB_DIR) -lcust_opapi
INC_COMMON = common
SRC_COMMON = common/common.cc
INCLUDEDIRS = -I$(HCCL_INC_DIR)\
				-I$(ACL_INC_DIR)/acllib/include \
		 		-I$(ACL_INC_DIR)/opp/include/ \
				-I$(ACL_INC_DIR)/opp/vendors/customize/op_api/include/ \
		 		-I$(ACL_INC_DIR)/aarch64-linux/include/ \
				-I$(INC_COMMON)/ \
				-I$(MPI_INC_DIR)

ifeq ($(shell uname -m), aarch64)
$(info "aarch64 verison")
ARCH := aarch
else
$(info "x86 verison")
ARCH := x86
endif
#
#make
#--------------------------------------------------------------------------------------------------------------------------------------------------
mc2_test_obp:
	@mkdir -p bin
	#治水分支 主线未合入
	#g++ $(CXXFLAGS) ./obp/aclnnAllGatherQuantBatchMatMul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ./bin/aclnnAllGatherQuantBatchMatMul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/hccl_allreduce.cc $(INCLUDEDIRS) -o ./bin/hccl_allreduce_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS) ./obp/mm_allreduce.cc $(INCLUDEDIRS) -o ./bin/allreduce_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/mm_allreduce_arn.cc $(INCLUDEDIRS) -o ./bin/mc2_arn_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnAllGatherMatmul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ./bin/aclnnAllGatherMatmul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnMatmulReduceScatter.cc $(INCLUDEDIRS) -o ./bin/aclnnMatmulReduceScatter_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnAlltoAllAllGatherBatchMatMul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ./bin/aclnnAlltoAllAllGatherBatchMatMul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnBatchMatMulReduceScatterAlltoAll.cc $(INCLUDEDIRS) -o ./bin/aclnnBatchMatMulReduceScatterAlltoAll_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnWeightQuantMatmulAllReduce.cc  $(INCLUDEDIRS) -o ./bin/aclnnWeightQuantMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnQuantMatmulAllReduce.cc $(INCLUDEDIRS) -o ./bin/aclnnQuantMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnQuantMatmulAllReduceV2.cc $(INCLUDEDIRS) -o ./bin/aclnnQuantMatmulAllReduceV2_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnQuantMatmulAllReduceV3.cc $(INCLUDEDIRS) -o ./bin/aclnnQuantMatmulAllReduceV3_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnMatmulAllReduce.cc $(INCLUDEDIRS) -o ./bin/aclnnMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnMatmulAllReduceV2.cc $(INCLUDEDIRS) -o ./bin/aclnnMatmulAllReduceV2_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/hccl_reducescatter.cc $(INCLUDEDIRS) -o ./bin/hccl_reducescatter_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/hccl_allgather.cc $(INCLUDEDIRS) -o ./bin/hccl_allgather_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnAlltoAllvGroupedMatMul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ./bin/aclnnAlltoAllvGroupedMatMul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./obp/aclnnGroupedMatMulAlltoAllv.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ./bin/aclnnGroupedMatMulAlltoAllv_$(ARCH) $(CUSTOM_LIBS)
	@printf "\033[0;32;32 allreduce_uquant  compile completed\n\033[m"
mc2_test_david:
	@mkdir -p bin
	g++ $(CXXFLAGS)  ./david/aclnnQuantAllReduce.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantAllReduce_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnQuantReduceScatter.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantReduceScatter_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnAllGatherMatmul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnAllGatherMatmul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnMatmulAllReduce.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnAllGatherMatmulV2.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnAllGatherMatmulV2_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnMatmulAllReduceV2.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnMatmulAllReduceV2_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnMatmulReduceScatterV2.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnMatmulReduceScatterV2_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnQuantMatmulAllReduce.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnQuantMatmulAllReduceV2.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantMatmulAllReduceV2_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnQuantMatmulAllReduceV3.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantMatmulAllReduceV3_$(ARCH) $(CUSTOM_LIBS)
	#g++ $(CXXFLAGS)  ./david/aclnnQuantMatmulAllReduceV4.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantMatmulAllReduceV4_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnWeightQuantMatmulAllReduce.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnWeightQuantMatmulAllReduce_$(ARCH) $(CUSTOM_LIBS)
#	g++ $(CXXFLAGS)  ./david/aclnnAlltoAllMatmul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnAlltoAllMatmul_$(ARCH) $(CUSTOM_LIBS)
#	g++ $(CXXFLAGS)  ./david/aclnnMatmulAlltoAll.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnMatmulAlltoAll_$(ARCH) $(CUSTOM_LIBS)
#	g++ $(CXXFLAGS)  ./david/aclnnAlltoAllQuantMatmul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnAlltoAllQuantMatmul_$(ARCH) $(CUSTOM_LIBS)
#	g++ $(CXXFLAGS)  ./david/aclnnQuantMatmulAlltoAll.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantMatmulAlltoAll_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnAlltoAllvQuantGroupedMatMul.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnAlltoAllvQuantGroupedMatMul_$(ARCH) $(CUSTOM_LIBS)
	g++ $(CXXFLAGS)  ./david/aclnnQuantGroupedMatMulAlltoAllv.cc $(SRC_COMMON) $(INCLUDEDIRS) -o ../bin_path/aclnnQuantGroupedMatMulAlltoAllv_$(ARCH) $(CUSTOM_LIBS)

	@printf "\033[0;32;32 allreduce_uquant  compile completed\n\033[m"

.PHONY: clean
clean:
	rm -rf hccl_allreduce mm_allreduce
