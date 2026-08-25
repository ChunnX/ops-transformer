#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_pa_kv_cache_with_dequant.h"
using namespace std;
int64_t Sz(const vector<int64_t>&s){int64_t r=1;for(auto i:s)r*=i;return r;}
template<typename T> int CT(const vector<T>&d,const vector<int64_t>&sh,void**da,aclDataType dt,aclTensor**t){
    auto sz=Sz(sh)*sizeof(T);if(aclrtMalloc(da,sz,ACL_MEM_MALLOC_HUGE_FIRST))return 1;
    if(aclrtMemcpy(*da,sz,d.data(),sz,ACL_MEMCPY_HOST_TO_DEVICE))return 1;
    vector<int64_t>st(sh.size(),1);for(int64_t i=sh.size()-2;i>=0;i--)st[i]=sh[i+1]*st[i+1];
    *t=aclCreateTensor(sh.data(),sh.size(),dt,st.data(),0,ACL_FORMAT_ND,sh.data(),sh.size(),*da);return 0;}
int main(){int r=0;
    const int nb=2,bs=32,nh=2,hd=64,nt=64;
    int32_t dev=0;aclrtStream st;void*wa=0,*d[8]={0};aclTensor*t[8]={0};uint64_t ws=0;aclOpExecutor*ex=0;
    vector<uint16_t> ko(8192),vo(8192);
    { // protect goto
    if(aclInit(nullptr)){r=1;goto out;}if(aclrtSetDevice(dev)){r=2;goto out;}if(aclrtCreateStream(&st)){r=3;goto out;}
    vector<int8_t> kcH(nb*bs*nh*hd);for(size_t i=0;i<kcH.size();i++)kcH[i]=(i%255)-127;
    vector<uint8_t> vcH(nb*bs*nh*hd);for(size_t i=0;i<vcH.size();i++)vcH[i]=i%128;
    vector<float> ksH(nb*bs*nh,1.0f),vsH(nh*hd,0.5f);
    vector<int32_t> btH={0,1},slH={64};vector<uint16_t> kH(nt*nh*hd,0),vH(nt*nh*hd,0);
    CT(kcH,{nb,bs,nh,hd},&d[0],ACL_INT8,&t[0]);CT(vcH,{nb,bs,nh,hd},&d[1],ACL_INT8,&t[1]);
    CT(ksH,{nb,bs,nh},&d[2],ACL_FLOAT,&t[2]);CT(vsH,{nh,hd},&d[3],ACL_FLOAT,&t[3]);
    CT(btH,{1,2},&d[4],ACL_INT32,&t[4]);CT(slH,{1},&d[5],ACL_INT32,&t[5]);
    CT(kH,{nt,nh,hd},&d[6],ACL_BF16,&t[6]);CT(vH,{nt,nh,hd},&d[7],ACL_BF16,&t[7]);
    r=aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],nullptr,false,&ws,&ex);
    if(r){cerr<<"GetWS "<<r<<"\n";goto out;}
    cout<<"WS="<<ws<<endl;if(ws>0)aclrtMalloc(&wa,ws,ACL_MEM_MALLOC_HUGE_FIRST);
    r=aclnnGatherPaKvCacheWithDequant(wa,ws,ex,st);if(r){cerr<<"Exec "<<r<<"\n";goto out;}
    aclrtSynchronizeStream(st);
    aclrtMemcpy(ko.data(),ko.size()*2,d[6],ko.size()*2,ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(vo.data(),vo.size()*2,d[7],vo.size()*2,ACL_MEMCPY_DEVICE_TO_HOST);
    cout<<"Key[0..4]: ";for(int i=0;i<5;i++)cout<<ko[i]<<" ";cout<<"\n";
    cout<<"Val[0..4]: ";for(int i=0;i<5;i++)cout<<vo[i]<<" ";cout<<"\n";
    cout<<"E2E PASSED"<<endl;
    }
out:if(wa)aclrtFree(wa);for(int i=0;i<8;i++){if(t[i])aclDestroyTensor(t[i]);if(d[i])aclrtFree(d[i]);}
    aclrtDestroyStream(st);aclrtResetDevice(dev);aclFinalize();return r;}
