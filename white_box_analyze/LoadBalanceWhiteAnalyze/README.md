1、不支持v1接口
2、mBasizeSize,s2BaseSize计算逻辑较为繁琐，计算逻辑待后续更新，改版本暂不考虑
覆盖率结果编码解释：
s1g:0为gs1格式，1为s1g
batch_actualseq:0为没有空batch,8为全0batch，"xxx"结果表示空ensor情况，第一位1表示头为空batch，第二位1表示中间有空batch，最后一位1表示尾部有空batch
sparemode:sparsemode情况
n2:n2覆盖情况
g:g覆盖情况

【v2 版本更新】
WhiteBox_cal_MS2_basesize_v2.py
支持计算mbasesize和s2basesize
通过mbasesize和s2basesize的值计算行和列的尾块情况
(mBaseSize,s2BaseSize)编码结果解释：
将编码转为二进制理解，第一位表示行切分情况，第二位表示列切分情况
0->00:行和列切分都无尾块
1->01：行切分无尾块，列切分有尾块
2->10：行切分有尾块，列切分无尾块
3->11：行切分和列切分都有尾块