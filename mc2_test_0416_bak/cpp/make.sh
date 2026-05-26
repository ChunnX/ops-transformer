source /usr/local/Ascend/latest/bin/setenv.bash
export ASCEND_DIR=/usr/local/Ascend/latest

if [ "$1" == "david" ];then
echo " make david ...."
make -j32 mc2_test_david  SOC_VERSON=910D
else
echo " make obp ...."
make -j32 mc2_test_obp
fi

