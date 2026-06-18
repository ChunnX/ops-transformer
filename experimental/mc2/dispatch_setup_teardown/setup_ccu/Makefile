# 独立构建 CCU dispatch_setup 算子 .so —— 完全在 experimental/mc2 下编, 不依赖 hccl/build.sh。
# 配方照搬官方 CCU 自定义算子样例的 CMakeLists(已验证可编)。
ifndef ASCEND_HOME_PATH
$(error please run: source /home/0617/env/set_env.sh)
endif

A := $(ASCEND_HOME_PATH)
CXX := g++

CXXFLAGS := -U_FORTIFY_SOURCE -Werror -fno-common -fno-strict-aliasing -pipe -g -O0 \
            -std=c++14 -fstack-protector-all -fPIC \
            -D_GLIBCXX_USE_CXX11_ABI=0 -DHOST_COMPILE

INCS := -Iinc -Iop_host -Iop_kernel_ccu \
        -I$(A)/include -I$(A)/include/hccl -I$(A)/include/hcomm -I$(A)/include/hcomm/ccu \
        -I$(A)/pkg_inc -I$(A)/pkg_inc/hcomm -I$(A)/pkg_inc/hcomm/ccu

SRCS := op_host/dispatch_setup.cc op_host/utils.cc \
        op_kernel_ccu/ccu_kernel.cc op_kernel_ccu/exec_op.cc

LDFLAGS := -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -L$(A)/lib64
LIBS := -lhcomm -lc_sec -lascendcl

TARGET := libdispatch_setup_ccu.so

all:
	$(CXX) $(CXXFLAGS) -shared $(INCS) $(SRCS) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@echo "[OK] built $(TARGET) (CCU dispatch_setup, 全程在 experimental/mc2 下)"

clean:
	rm -f $(TARGET)
