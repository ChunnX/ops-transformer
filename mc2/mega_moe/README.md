# MegaMoE

## 浜у搧鏀寔鎯呭喌

| 浜у搧                                                         |  鏄惁鏀寔   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    鈭?   |
| <term>Atlas A3 璁粌绯诲垪浜у搧/Atlas A3 鎺ㄧ悊绯诲垪浜у搧</term>       |    鈭?   |
| <term>Atlas A2 璁粌绯诲垪浜у搧/Atlas A2 鎺ㄧ悊绯诲垪浜у搧</term> |    鈭?   |
| <term>Atlas 200I/500 A2 鎺ㄧ悊浜у搧</term>                      |    脳    |
| <term>Atlas 鎺ㄧ悊绯诲垪浜у搧</term>                               |    脳    |
| <term>Atlas 璁粌绯诲垪浜у搧</term>                              |    脳    |

## 鍔熻兘璇存槑

- 绠楀瓙鍔熻兘锛歁egaMoE绠楀瓙 灏?MoE 灞傜殑涓撳 FFN 鐨勫畬鏁磋绠楁祦绋嬪強鍓嶅悗鏁版嵁閫氫俊锛堝嵆 Dispatch + Linear1 + SwiGLU + Linear2 + Combine锛夎瀺鍚堜负鍗曚釜绠楀瓙锛屽疄鐜颁簡閫氫俊鍜岃绠楃殑鎺╃洊銆?
- 璁＄畻鍏紡锛?  - 杈撳叆锛?    - $\mathbf{X} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$锛氭縺娲荤煩闃碉紝瀵瑰簲鍏ュ弬 `x`銆?\text{totalNumTokens}$ 鏄叏灞€鎬?token 鏁帮紝$\text{hidden}$ 鏄殣钘忓眰缁村害銆?    - $\mathbf{E} \in \mathbb{Z}^{\text{totalNumTokens} \times \text{topK}}$锛歵oken 閫夋嫨鐨勪笓瀹剁紪鍙风煩闃碉紝瀵瑰簲鍏ュ弬 `topkIds`銆?\text{topK}$ 鏄瘡涓?token 閫夋嫨鐨勪笓瀹舵暟閲忋€?    - $\mathbf{G} \in \mathbb{R}^{\text{totalNumTokens} \times \text{topK}}$锛歵oken 閫夋嫨鐨勪笓瀹剁殑闂ㄦ帶鏉冮噸鐭╅樀锛屽搴斿叆鍙?`topkWeights`銆?    - $\mathbf{W}_1^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{hidden} \times (2 \text{intermediateHidden})}$锛氳矾鐢?MoE 涓撳鐨?Linear1 鏉冮噸锛屽搴斿叆鍙?`weight1` 鐨?MoE 涓撳閮ㄥ垎銆?    - $\mathbf{W}_2^{\mathrm{moe}} \in \mathbb{R}^{\text{localMoeExpertNum} \times \text{intermediateHidden} \times \text{hidden}}$锛氳矾鐢?MoE 涓撳鐨?Linear2 鏉冮噸锛屽搴斿叆鍙?`weight2` 鐨?MoE 涓撳閮ㄥ垎銆?    - $\mathbf{W}_1^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{hidden} \times (2 \text{intermediateHidden})}$锛氬叡浜笓瀹剁殑 Linear1 鏉冮噸锛屽搴斿叆鍙?`sharedWeight1`銆?    - $\mathbf{W}_2^{\mathrm{shared}} \in \mathbb{R}^{\text{sharedExpertNumPerRank} \times \text{intermediateHidden} \times \text{hidden}}$锛氬叡浜笓瀹剁殑 Linear2 鏉冮噸锛屽搴斿叆鍙?`sharedWeight2`銆?  - 杈撳嚭锛?
    - $\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$锛氭渶缁堣緭鍑虹煩闃碉紝瀵瑰簲鍑哄弬 `y`銆?  - 绾﹀畾锛?    - $鈰? 琛ㄧず鐭╅樀涔樻硶锛?鈯? 琛ㄧず閫愬厓绱犱箻娉曘€?    - $\left \lfloor z\right \rceil$ 琛ㄧず灏?$z$ 鍥涜垗浜斿叆鍒版渶杩戠殑鏁存暟锛?\left \lfloor z\right \rfloor$ 琛ㄧず灏?$z$ 鍚戜笅鍙栨暣銆?    - $|z|$ 琛ㄧず鍙栫粷瀵瑰€硷紝$\max(z)$ 琛ㄧず鍙栨渶澶у€笺€?    - 鍏ㄤ綋 token 鐨勯泦鍚堜负 $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{totalNumTokens} - 1\} \}$銆?    - $\text{token}_i$ 鐨?token 琛ㄧず锛堝嵆闅愯棌鐘舵€佸悜閲忥級涓?$\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$锛屼笖 $\mathbf{x}_i = \mathbf{X}[i,:]$銆?    - $\text{token}_i$ 鐨勪笓瀹剁储寮曚负 $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{topK} - 1\},\quad e_{i,k} \in \{0,\dots,\text{moeExpertNum} - 1\}$銆?    - 婵€娲诲嚱鏁?$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}$锛屽叾涓?$\sigma$ 涓?Sigmoid 鍑芥暟銆?    - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$銆傚叾涓?$\mathbb{Z}_8^{\text{sym}}$ 鐨勪笂鏍?$\text{sym}$ 琛ㄧず瀵圭О閲忓寲鍊煎煙鍖洪棿锛氬叾鍊煎煙鍏充簬 $-127$ 涓?$127$ 瀵圭О鍙栨暣锛屼笌鏍囧噯 INT8 鐨?$[-128, 127]$ 鍊煎煙涓嶅悓锛屾晠浠?$\text{sym}$ 涓婃爣鍖哄垎銆?    - 寮犻噺鍒囩墖鎿嶄綔閲囩敤Python椋庢牸鐨?`start:stop:step` 琛ㄧず娉曪紝渚嬪$[0::2, :]$ 浠ｈ〃鍙栧伓鏁拌銆?[1::2, :]$ 浠ｈ〃鍙栧鏁拌銆?    - $\mathrm{bitcast}_{T}(\mathbf{Z})$ 琛ㄧず浜岃繘鍒堕噸瑙ｉ噴鎿嶄綔锛屽皢寮犻噺$\mathbf{Z}$鐨勫簳灞備簩杩涘埗鏁版嵁鎸夌洰鏍囩被鍨?$T$ 閲嶆柊瑙ｉ噴銆?- 璁＄畻杩囩▼

    鍚勪骇鍝佹敮鎸佺殑**Linear璁＄畻鏃?*鐨勬縺娲荤煩闃礎鍜屾潈閲嶇煩闃礧鐨勬暟鎹被鍨嬪涓嬶細

    | 鍦烘櫙鍚?   |  A |  W   |
    | --- | :---:  | :---:        |
    | A16W16   | BFLOAT16     |BFLOAT16        |
    | A8W8-INT | INT8   | INT8         |
    | A8W8-FP  | FLOAT8_E4M3FN銆丗LOAT8_E5M2 |FLOAT8_E4M3FN銆丗LOAT8_E5M2 |
    | A8W4-INT | INT8        | INT4            |
    | A8W4-FP | FLOAT8_E4M3FN        | FLOAT4_E2M1          |
    | A4W4-FP | FLOAT4_E2M1        | FLOAT4_E2M1            |

    - <term>Atlas A2 璁粌绯诲垪浜у搧/Atlas A2 鎺ㄧ悊绯诲垪浜у搧</term>銆?term>Atlas A3 璁粌绯诲垪浜у搧/Atlas A3 鎺ㄧ悊绯诲垪浜у搧</term>锛氫笉鏀寔涓婅〃涓瑼8W8-FP鍦烘櫙銆?    - <term>Ascend 950PR/Ascend 950DT</term>锛氫笉鏀寔涓婅〃涓瑼16W16 銆丄8W8-INT銆丄8W4-INT鍦烘櫙銆?
    <details>
    <summary> A16W16 闈為噺鍖栧満鏅?/summary>

    - **EP Dispatch**

        鍦?Dispatch 闃舵锛屾瘡涓?$\text{token}_i$ 灏嗗叾token琛ㄧず $\mathbf{x}_i$ 鍙戦€佺粰涓撳 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$銆傚嵆瀵逛簬姣忎釜 $k$锛屼笓瀹?$e_{i,k}$ 鎺ユ敹涓€浠?$\mathbf{x}_i$銆?
        璁?$I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 涓烘墍鏈夎鍒嗘淳缁欎笓瀹?$e$ 鐨?token 绱㈠紩闆嗗悎锛岄泦鍚?$I_e$ 鐨勫ぇ灏?$N_e = |I_e|$ 鍗充负涓撳 $e$ 闇€瑕佸鐞嗙殑 token 鎬绘暟锛屽垯鏈?$\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 鏄敱鎵€鏈夋弧瓒?$i \in I_e$ 鐨則oken琛ㄧず $\mathbf{x}_i$ **鎸変换鎰忓浐瀹氶『搴忚鍫嗗彔**鑰屾垚鐨勭煩闃碉紝璇ョ煩闃靛嵆涓轰笓瀹?$e$ 缁忚繃 Dispatch 鍚庢敹鍒扮殑鍏ㄩ儴 token 琛ㄧず銆?
        瀵逛簬姣忎釜 $\text{token}_i$ 鍙婂叾閫変腑鐨勭 $k$ 涓笓瀹?$e_{i,k}$锛屽瓨鍦ㄥ敮涓€鐨勮绱㈠紩 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$锛屼娇寰?$\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$銆傝鏄犲皠璁板綍浜?$\mathbf{x}_i$ 鍦ㄤ笓瀹?$e_{i,k}$ 鐨勮緭鍏ョ煩闃?$\mathbf{X}_{e_{i,k}}$ 涓殑浣嶇疆銆?
    - **Expert Compute**

        鍦?MoE 灞備腑锛屾瘡涓笓瀹舵湰璐ㄤ笂鏄竴涓嫭绔嬬殑鍓嶉缃戠粶锛團FN锛夛紝閲囩敤 **SwiGLU** 缁撴瀯浠ユ彁鍗囪〃杈捐兘鍔涖€傛暣涓绠楄繃绋嬪垎涓哄涓嬩笁涓瓙姝ラ銆?
        **1. Linear1 鎶曞奖**

        Linear1 鎶曞奖鏄笓瀹剁綉缁滅殑绗竴灞傜嚎鎬у彉鎹紝鍚屾椂浜х敓 **gate 閮ㄥ垎** 鍜?**up 閮ㄥ垎** 鎵€闇€鐨勯婵€娲诲€笺€傚叾璁＄畻鍏紡涓?        $$
        \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}\bigr)
        $$
        **2. SwiGLU 婵€娲?*

        棣栧厛灏?$\mathbf{H}_e$ 娌垮垪缁村害鎷嗗垎涓?**gate 閮ㄥ垎** $\mathbf{H}_{\text{gate}}$ 鍜?**up 閮ㄥ垎** $\mathbf{H}_{\text{up}}$锛岀劧鍚庡 gate 閮ㄥ垎搴旂敤 SiLU 婵€娲诲嚱鏁帮紝鍐嶄笌 up 閮ㄥ垎閫愬厓绱犵浉涔橈紝寰楀埌涓撳 $e$ 鐨勪腑闂存縺娲昏〃绀?$\mathbf{A}_e$銆?        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e\bigl[:,\;\text{:intermediateHidden}\bigr], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e\bigl[:,\;\text{intermediateHidden:}\bigr]
        $$
        $$
        \mathbf{A}_e = \text{SiLU}\bigl(\mathbf{H}_{\text{gate}}\bigr) \;\odot\; \mathbf{H}_{\text{up}}
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{intermediateHidden}}\bigr)
        $$

        **3. Linear2 鎶曞奖**

        Linear2 鎶曞奖浣滀负绗簩灞傜嚎鎬у彉鎹紝灏嗕腑闂存縺娲昏〃绀?$\mathbf{A}_e$ 浠庨珮缁寸┖闂存姇褰卞洖鍘熷鐨勯殣钘忕淮搴?$\text{hidden}$锛屼娇涓撳杈撳嚭鑳藉涓庢畫宸繛鎺ョ瓑鍚庣画鎿嶄綔鍏煎銆?        $$
        \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
        \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
        $$

        缁忚繃浠ヤ笂璁＄畻锛屼笓瀹?$e$ 鐨勬瘡涓€琛岃緭鍑哄搴斿叾鎵规涓殑涓€涓?token銆傚浜?$\text{token}_i$锛屽畠鍦ㄤ笓瀹?$e_{i,k}$ 涓殑杈撳嚭琛屽嵆涓?$\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$銆?    - **Token Combine**

        Combine 璐熻矗鏀堕泦鎵€鏈変笓瀹惰绠楀嚭鐨勮緭鍑哄悜閲忥紝鎸夌収姣忎釜 token 鍘熷厛鍒嗛厤鍒扮殑涓撳鏉冮噸杩涜鍔犳潈姹傚拰锛屾渶缁堜负姣忎釜 token 鐢熸垚涓€涓瀺鍚堝悗鐨勮緭鍑恒€傚埄鐢ㄤ箣鍓嶈褰曠殑浣嶇疆绱㈠紩 $\operatorname{row}(i,k)$锛屼粠涓撳 $e_{i,k}$ 鐨勮緭鍑虹煩闃典腑鏀跺洖灞炰簬 $\text{token}_i$ 鐨勮锛屽苟涓庨棬鎺ф潈閲嶇浉涔樺悗姹傚拰锛?
        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\;
        \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$

        鍏朵腑 $w_k = \mathbf{G}[i,k]$ 涓?$\text{token}_i$ 瀵逛笓瀹?$e_{i,k}$ 鐨勯棬鎺ф潈閲嶃€?
        鎵€鏈?token 鐨勮緭鍑烘寜杈撳叆椤哄簭鍫嗗彔涓烘渶缁堣緭鍑?$\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$銆?    </details>

    <details>
    <summary> A8W8-INT 閲忓寲鍦烘櫙</summary>

    - **杈撳叆**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$锛歀inear1 鏉冮噸鐭╅樀鐨勯€愰€氶亾缂╂斁鍥犲瓙锛屽搴斿叆鍙?`weightScales1`銆?        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$锛歀inear2 鏉冮噸鐭╅樀鐨勯€愰€氶亾缂╂斁鍥犲瓙锛屽搴斿叆鍙?`weightScales2`銆?
    - **EP Dispatch**

        鍦?Dispatch 閫氫俊涔嬪墠锛岄鍏堝皢鍘熷 BF16 婵€娲荤煩闃?$\mathbf{X}$ 閲忓寲涓?INT8銆傚姣忎釜 $\text{token}_i$锛岃绠楀叾閫?token 缂╂斁鍥犲瓙锛?        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        鐒跺悗閲忓寲寰楀埌 INT8 琛ㄧず锛?        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        鍦?Dispatch 閫氫俊闃舵锛屾瘡涓?$\text{token}_i$ 灏嗗叾閲忓寲鍚庣殑鍚戦噺 $\mathbf{q}_i$ 鍜岀缉鏀惧洜瀛?$s^{X}_i$ 鍙戦€佺粰涓撳 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$锛?e_{i,k} = \mathbf{E}[i,k]$锛夈€?
        璁?$I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 涓烘墍鏈夎鍒嗘淳缁欎笓瀹?$e$ 鐨?token 绱㈠紩闆嗗悎锛岄泦鍚?$I_e$ 鐨勫ぇ灏?$N_e = |I_e|$ 鍗充负涓撳 $e$ 闇€瑕佸鐞嗙殑 token 鎬绘暟銆傚垯鏈?$\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 鏄敱鎵€鏈夋弧瓒?$i \in I_e$ 鐨?$\mathbf{q}_i$ **鎸変换鎰忓浐瀹氶『搴忚鍫嗗彔**鑰屾垚鐨勭煩闃碉紝璇ョ煩闃靛嵆涓轰笓瀹?$e$ 缁忚繃 Dispatch 鍚庢敹鍒扮殑鍏ㄩ儴 token 琛ㄧず锛涘悓鐞嗭紝瀵瑰簲鐨勪笓瀹?$e$ 鏀跺埌鐨勭缉鏀惧洜瀛愬悜閲忚涓?$\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$锛屽叾鍏冪礌鐢辨墍鏈夋弧瓒?$i \in I_e$ 鐨?$s^{X}_i$ 鎸変笌 $\mathbf{Q}_e$ 鐩稿悓鐨勮椤哄簭鍫嗗彔鑰屾垚銆?
        瀵逛簬姣忎釜 $\text{token}_i$ 鍙婂叾閫変腑鐨勭 $k$ 涓笓瀹?$e_{i,k}$锛屽瓨鍦ㄥ敮涓€鐨勮绱㈠紩 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$锛屼娇寰?$\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$銆傝鏄犲皠璁板綍浜?$\mathbf{q}_i$ 鍦ㄤ笓瀹?$e_{i,k}$ 鐨勮緭鍏ョ煩闃典腑鐨勪綅缃€?
    - **Expert Compute**

        鍦?MoE 灞備腑锛屾瘡涓笓瀹舵湰璐ㄤ笂鏄竴涓嫭绔嬬殑鍓嶉缃戠粶锛團FN锛夛紝閲囩敤 SwiGLU 缁撴瀯浠ユ彁鍗囪〃杈捐兘鍔涖€傚湪A8W8鍦烘櫙涓嬶紝涓や釜绾挎€у眰閮戒娇鐢?INT8 杈撳叆鍜?INT8 鏉冮噸杩涜鐭╅樀涔橈紝寰楀埌 INT32 涓棿缁撴灉骞跺弽閲忓寲銆傚叿浣撳垎涓轰笁涓瓙姝ラ銆?
        **1. Linear1 鎶曞奖锛圛NT8 鐭╅樀涔?+ 鍙嶉噺鍖栵級**

        Linear1 鎶曞奖鏄笓瀹剁綉缁滅殑绗竴灞傜嚎鎬у彉鎹紝鍚屾椂浜х敓 **gate 閮ㄥ垎** 鍜?**up 閮ㄥ垎** 鎵€闇€鐨勯婵€娲诲€笺€傝绠楁椂鎵ц INT8 鐭╅樀涔樻硶锛屽緱鍒?INT32 璁＄畻缁撴灉锛?        $$
        \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$
        鐒跺悗鍙嶉噺鍖栦负棰勬縺娲诲€?$\mathbf{H}_e$锛?        $$
        \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **2. SwiGLU 婵€娲?*

        棣栧厛灏?$\mathbf{H}_e$ 娌垮垪缁村害鎷嗗垎涓?**gate 閮ㄥ垎** $\mathbf{H}_{\text{gate}}$ 鍜?**up 閮ㄥ垎** $\mathbf{H}_{\text{up}}$锛岀劧鍚庡 gate 閮ㄥ垎搴旂敤 SiLU 婵€娲诲嚱鏁帮紝鍐嶄笌 up 閮ㄥ垎閫愬厓绱犵浉涔橈紝寰楀埌涓撳 $e$ 鐨勪腑闂存縺娲昏〃绀?$\mathbf{A}_e$銆?        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:, :\text{intermediateHidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:, \text{intermediateHidden}:]
        $$
        $$
        \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
        $$

        **3. Linear2 鎶曞奖锛堥噺鍖?+ INT8 鐭╅樀涔?+ 鍙嶉噺鍖栵級**

        Linear2 鎶曞奖浣滀负绗簩灞傜嚎鎬у彉鎹紝灏嗕腑闂存縺娲昏〃绀?$\mathbf{A}_e$ 浠庨珮缁寸┖闂存姇褰卞洖鍘熷鐨勯殣钘忕淮搴?$\text{hidden}$锛屼娇涓撳杈撳嚭鑳藉涓庢畫宸繛鎺ョ瓑鍚庣画鎿嶄綔鍏煎銆?        鍦?A8W8 鍦烘櫙涓嬶紝闇€瑕佸皢婵€娲诲€奸噺鍖栦负 INT8锛屽洜姝ゅ厛瀵?$\mathbf{A}_e$ 鐨勬瘡涓€琛岋紙姣忎釜 token锛夎绠楃缉鏀惧洜瀛愶細
        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        寰楀埌涓撳 $e$ 鍦?Linear2 璁＄畻鏃剁殑婵€娲荤缉鏀惧洜瀛?$\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$銆傜劧鍚庨噺鍖栵細
        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$
        鍐嶆墽琛?INT8 鐭╅樀涔樻硶骞跺弽閲忓寲锛?        $$
        \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
        $$
        $$
        \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        缁忚繃浠ヤ笂璁＄畻锛屼笓瀹?$e$ 鐨勬瘡涓€琛岃緭鍑哄搴斿叾鎵规涓殑涓€涓?token銆傚浜?$\text{token}_i$锛屽畠鍦ㄤ笓瀹?$e_{i,k}$ 涓殑杈撳嚭琛屽嵆涓?$\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$銆?
    - **Token Combine**

        Combine 璐熻矗鏀堕泦鎵€鏈変笓瀹惰绠楀嚭鐨勮緭鍑哄悜閲忥紝鎸夌収姣忎釜 token 鍘熷厛鍒嗛厤鍒扮殑涓撳鏉冮噸杩涜鍔犳潈姹傚拰锛屾渶缁堜负姣忎釜 token 鐢熸垚涓€涓瀺鍚堝悗鐨勮緭鍑恒€傚埄鐢ㄤ箣鍓嶈褰曠殑浣嶇疆绱㈠紩 $\operatorname{row}(i,k)$锛屼粠涓撳 $e_{i,k}$ 鐨勮緭鍑虹煩闃典腑鏀跺洖灞炰簬 $\text{token}_i$ 鐨勮锛屽苟涓庨棬鎺ф潈閲嶇浉涔樺悗姹傚拰锛?        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
        $$
        鍏朵腑 $w_k = \mathbf{G}[i,k]$ 涓?$\text{token}_i$ 瀵逛笓瀹?$e_{i,k}$ 鐨勯棬鎺ф潈閲嶃€?
        鎵€鏈?token 鐨勮緭鍑烘寜杈撳叆椤哄簭鍫嗗彔涓烘渶缁堣緭鍑?$\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$銆?
    </details>

    <details>
    <summary> A8W4-INT 閲忓寲鍦烘櫙</summary>

    - **杈撳叆**
        - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$锛歀inear1 鏉冮噸鐭╅樀鐨勯€愰€氶亾缂╂斁鍥犲瓙锛屽搴斿叆鍙?`weightScales1`銆?        - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$锛歀inear2 鏉冮噸鐭╅樀鐨勯€愰€氶亾缂╂斁鍥犲瓙锛屽搴斿叆鍙?`weightScales2`銆?        - $\mathbf{B}_1 \in \mathbb{R}^{\text{moeExpertNum} \times (2\times \text{intermediateHidden})}$锛歀inear1 鐨勫亸缃煩闃碉紝鐢?INT4 閲忓寲杩囩▼绂荤嚎鐢熸垚锛屽搴斿叆鍙?`bias1`銆?        - $\mathbf{B}_2 \in \mathbb{R}^{\text{moeExpertNum} \times \text{hidden}}$锛歀inear2 鐨勫亸缃煩闃碉紝鐢?INT4 閲忓寲杩囩▼绂荤嚎鐢熸垚锛屽搴斿叆鍙?`bias2`銆?
    - **EP Dispatch**

        鍦?Dispatch 閫氫俊涔嬪墠锛岄鍏堝皢鍘熷 BF16 婵€娲荤煩闃?$\mathbf{X}$ 閲忓寲涓?INT8銆傚姣忎釜 $\text{token}_i$锛岃绠楀叾閫?token 缂╂斁鍥犲瓙锛?        $$
        s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
        $$
        鐒跺悗閲忓寲寰楀埌 INT8 琛ㄧず锛?        $$
        \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
        $$

        鍦?Dispatch 閫氫俊闃舵锛屾瘡涓?$\text{token}_i$ 灏嗗叾閲忓寲鍚庣殑鍚戦噺 $\mathbf{q}_i$ 鍜岀缉鏀惧洜瀛?$s^{X}_i$ 鍙戦€佺粰涓撳 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{topK}-1}$

        璁?$I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 涓烘墍鏈夎鍒嗘淳缁欎笓瀹?$e$ 鐨?token 绱㈠紩闆嗗悎锛岄泦鍚?$I_e$ 鐨勫ぇ灏?$N_e = |I_e|$ 鍗充负涓撳 $e$ 闇€瑕佸鐞嗙殑 token 鎬绘暟銆傚垯鏈?$\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 鏄敱鎵€鏈夋弧瓒?$i \in I_e$ 鐨?$\mathbf{q}_i$ **鎸変换鎰忓浐瀹氶『搴忚鍫嗗彔**鑰屾垚鐨勭煩闃碉紝璇ョ煩闃靛嵆涓轰笓瀹?$e$ 缁忚繃 Dispatch 鍚庢敹鍒扮殑鍏ㄩ儴 token 琛ㄧず锛涘悓鐞嗭紝瀵瑰簲鐨勪笓瀹?$e$ 鏀跺埌鐨勭缉鏀惧洜瀛愬悜閲忚涓?$\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$锛屽叾鍏冪礌鐢辨墍鏈夋弧瓒?$i \in I_e$ 鐨?$s^{X}_i$ 鎸変笌 $\mathbf{Q}_e$ 鐩稿悓鐨勮椤哄簭鍫嗗彔鑰屾垚銆?
        瀵逛簬姣忎釜 $\text{token}_i$ 鍙婂叾閫変腑鐨勭 $k$ 涓笓瀹?$e_{i,k}$锛屽瓨鍦ㄥ敮涓€鐨勮绱㈠紩 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$锛屼娇寰?$\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$銆傝鏄犲皠璁板綍浜?$\mathbf{q}_i$ 鍦ㄤ笓瀹?$e_{i,k}$ 鐨勮緭鍏ョ煩闃典腑鐨勪綅缃€?
    - **Expert Compute**

        鍦?MoE 灞備腑锛屾瘡涓笓瀹舵湰璐ㄤ笂鏄竴涓嫭绔嬬殑鍓嶉缃戠粶锛團FN锛夛紝閲囩敤 **SwiGLU** 缁撴瀯浠ユ彁鍗囪〃杈捐兘鍔涖€傚湪A8W4-INT鍦烘櫙涓嬶紝涓や釜绾挎€у眰閮介噰鐢∕SD锛圡ixed-precision Split-activation Decomposition锛屾贩鍚堢簿搴︽縺娲绘媶鍒嗗垎瑙ｏ級鏂规杩涜鐭╅樀涔橈紝閫氳繃灏?INT8 鍊兼媶涓洪珮4浣嶅拰浣?浣嶄袱涓湁绗﹀彿 INT4锛屼娇寰?INT8脳INT4 鐨勭煩闃典箻鍙垎瑙ｄ负涓や釜 INT4脳INT4 鐨勭煩闃典箻锛屼粠鑰屽埄鐢ㄧ‖浠剁殑 INT4 鐭╅樀涔樺姞閫熴€傝鏂规鐨勬暟瀛﹀師鐞嗗拰瀹炵幇閫昏緫鍙弬闃匸GroupedMatmul W4A8閲忓寲涓嶮SD鏂规](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81)銆?
        **1. 鐢熸垚绮惧害琛ュ伩鐨勫亸缃煩闃碉紙绂荤嚎鐢熸垚锛屽湪绠楀瓙澶栧畬鎴愶紝骞朵綔涓虹畻瀛愯緭鍏ワ級**

        MSD 鏂规鐨勬牳蹇冩槸灏嗛噺鍖栧悗鐨?INT8 婵€娲诲€间簩杩涘埗閲嶈В閲婂湴鎷嗗垎涓轰袱涓?INT4 鍒嗛噺銆傜敱浜庝綆浣?INT4 鍒嗛噺鎸?$(\mathbf{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$ 瀹氫箟锛屽叾鏁板€艰寖鍥磋鏄犲皠鍒?$[-8, 7]$锛岃繖鐩稿綋浜庡湪鍘熷鏃犵鍙蜂綆 4 浣嶏紙0~15锛夌殑鍩虹涓婂噺鍘讳簡 8銆傚綋灏嗛珮銆佷綆浣嶇殑 INT4 鍒嗗埆涓庢潈閲嶇煩闃靛仛鐭╅樀涔樺苟鍚堝苟鏃讹紝璇ュ亸绉讳細寮曞叆涓€涓父鏁伴」锛岄渶瑕佸湪鏈€缁堢粨鏋滀腑浜堜互琛ュ伩銆?
        璁板師濮?INT8 婵€娲荤煩闃典负 $\mathbf{X}^{\text{int8}}$锛屾媶鍒嗗悗鐨勯珮浣嶅拰浣庝綅 INT4 婵€娲荤煩闃靛垎鍒负 $\mathbf{X}_1^{\text{int4}}$ 鍜?$\mathbf{X}_2^{\text{int4}}$锛屾潈閲嶇煩闃典负 $\mathbf{W}$銆傛仮澶嶅叧绯讳负锛?
        $$
        \mathbf{X}^{\text{int8}} = 16 \times \mathbf{X}_1^{\text{int4}} + (\mathbf{X}_2^{\text{int4}} + 8 \cdot \mathbf{1}_{\text{mat}})
        $$

        鍏朵腑 $\mathbf{1}_{\text{mat}}$ 涓轰笌 $\mathbf{X}_2^{\text{int4}}$ 褰㈢姸鐩稿悓鐨勫叏 1 鐭╅樀锛岀敤浜庨€愬厓绱犲姞 8銆備护 $\mathbf{1}$ 涓哄舰鐘朵笌杈撳叆鐗瑰緛缁村害涓€鑷寸殑鍏?1 鍒楀悜閲忥紝鍒欑煩闃典箻灞曞紑涓猴細

        $$
        \mathbf{X}^{\text{int8}} \cdot \mathbf{W} = 16 \cdot (\mathbf{X}_1^{\text{int4}} \cdot \mathbf{W}) + (\mathbf{X}_2^{\text{int4}} \cdot \mathbf{W}) + 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W})
        $$

        杩欓噷 $\mathbf{1}^\top \cdot \mathbf{W}$ 琛ㄧず瀵规潈閲嶇煩闃?$\mathbf{W}$ 鐨勬瘡涓€鍒楁眰鍜岋紝鍏剁粨鏋滀负涓€涓鍚戦噺锛堝舰鐘朵笌杈撳嚭缁村害鐩稿悓锛夈€傚湪鍏蜂綋瀹炵幇涓紝璇ヨ鍚戦噺浼氳骞挎挱鍒版壒娆＄淮搴︼紝鍔犲埌鎵€鏈?token 鐨勮緭鍑轰笂銆傚彲瑙侊紝鑻ュ彧璁＄畻鍓嶄袱椤癸紝浼氱己澶变竴涓鏁伴」銆備负浜嗙簿纭仮澶嶇粨鏋滐紝闇€棰勫厛璁＄畻琛ュ伩鍋忕疆锛?
        $$
        \text{bias} = 8 \cdot (\mathbf{1}^\top \cdot \mathbf{W}),
        $$

        $\mathbf{B}_1$ 鍜?$\mathbf{B}_2$ 鍧囨寜姝ゆ硶锛屽垎鍒娇鐢ㄥ搴旂殑鍏?1 鍒楀悜閲忥紙缁村害鍒嗗埆涓?$\text{hidden}$ 鍜?$\text{intermediateHidden}$锛変笌鍚勮嚜鐨勬潈閲嶇煩闃电浉涔橈紝寰楀埌褰㈢姸鍖归厤鐨勫亸缃鍚戦噺銆傝繖浜涘亸缃湪绂荤嚎闃舵璁＄畻瀹屾垚鍚庝紶鍏ョ畻瀛愶紝鍦ㄥ悎骞堕珮浣庝綅缁撴灉鏃剁洿鎺ュ弬涓庡姞娉曪紝浠庤€屾秷闄ゅ亸绉诲紩鍏ョ殑璇樊銆?
        **2. Linear1 鎶曞奖锛圛NT4 脳 INT4 鐭╅樀涔?+ 鍙嶉噺鍖栵級**

        Linear1 鎶曞奖鏄笓瀹剁綉缁滅殑绗竴灞傜嚎鎬у彉鎹紝鍚屾椂浜х敓 **gate 閮ㄥ垎** 鍜?**up 閮ㄥ垎** 鎵€闇€鐨勯婵€娲诲€笺€?
        **(1) 婵€娲婚噸瑙ｉ噴涓篒NT4**

        灏?INT8 婵€娲诲紶閲?$\mathbf{Q}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 浜岃繘鍒堕噸瑙ｉ噴涓轰袱涓?INT4 浜ゆ浛鎷兼帴鐨勮鍥撅細

        $$
        \mathbf{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{hidden}}} \left( \mathbf{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{hidden}}
        $$

        鍏朵腑楂樹綅鍜屼綆浣嶅垎閲忕洿鎺ョ敱 $\mathbf{Q}_e^{\text{int4}}$ 鐨勫伓鏁拌鍜屽鏁拌缁欏嚭锛?
        $$
        \mathbf{Q}_e^{\text{high}} = \mathbf{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}, \quad
        \mathbf{Q}_e^{\text{low}} = \mathbf{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{hidden}}
        $$

        瀹冧滑鍦ㄦ暟鍊间笂鍙互鐢卞師 INT8 缁忓涓嬭绠楀緱鍒帮細

        $$
        \mathbf{Q}_e^{\text{high}} = \left\lfloor \frac{\mathbf{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{Q}_e^{\text{low}} = (\mathbf{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        鎭㈠鍏崇郴涓?$\mathbf{Q}_e^{\text{int8}} = 16\mathbf{Q}_e^{\text{high}} + (\mathbf{Q}_e^{\text{low}} + 8)$銆傜敱浜?$\mathrm{bitcast}$ 浠呮敼鍙樼被鍨嬭鍥撅紝$\mathbf{Q}_e^{\text{int4}}$ 涓?$\mathbf{Q}_e^{\text{int8}}$ 鍏变韩搴曞眰鐗╃悊鍐呭瓨锛屾棤闇€浠讳綍鏁版嵁閲嶆帓鎴栨嫹璐濄€?
        **(2) INT4 脳 INT4 鐭╅樀涔樹笌鏉冮噸鍙嶉噺鍖?*

        灏嗛噸瑙ｉ噴鍚庣殑 INT4 婵€娲昏鍥?$\mathbf{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{hidden}}$ 涓庢潈閲嶇煩闃?$\mathbf{W}_1[e] \in \mathbb{R}^{\text{hidden} \times 2\cdot\text{intermediateHidden}}$ 鎵ц鐭╅樀涔橈紝骞跺簲鐢ㄦ潈閲嶇缉鏀惧洜瀛?$\mathbf{s}^{W1}_e$ 杩涜鍙嶉噺鍖栵紝寰楀埌缁撴灉锛?
        $$
        \mathbf{C}_e = \bigl( \mathbf{Q}_e^{\text{int4}} \cdot \mathbf{W}_1[e] \bigr) \odot \mathbf{s}^{W1}_e
        \;\in\; \mathbb{R}^{2N_e \times 2\cdot\text{intermediateHidden}}
        $$

        灏嗙粨鏋滄寜鍋舵暟琛屽拰濂囨暟琛屽垎鍒涓哄涓嬬殑鐭╅樀瑙嗗浘锛屽畠浠嵆瀵瑰簲浜?\mathbf{Q}_e^{\text{high}}$鍜?\mathbf{Q}_e^{\text{low}}$鐨勮绠楃粨鏋滐細

        $$
        \mathbf{C}_e^{\text{high}} = \mathbf{C}_e[0::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}, \qquad
        \mathbf{C}_e^{\text{low}}  = \mathbf{C}_e[1::2, :] \in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        **(3) 绮惧害琛ュ伩涓庢縺娲诲弽閲忓寲**

        鍒╃敤鍋忕疆 $\mathbf{B}_1[e]$ 鍜屾縺娲荤缉鏀惧洜瀛?$\mathbf{s}^{X}_e$锛堣鍚戦噺锛夊垎鍒繘琛岀簿搴﹁ˉ鍋垮拰婵€娲诲弽閲忓寲锛屾渶缁堥婵€娲诲€间负锛?
        $$
        \mathbf{H}_e = \Bigl( 16 \cdot \mathbf{C}_e^{\text{high}} + \mathbf{C}_e^{\text{low}} + \mathbf{B}_1[e] \Bigr) \odot \mathbf{s}^{X}_e \quad\in \mathbb{R}^{N_e \times 2\cdot\text{intermediateHidden}}
        $$

        鍏朵腑 $\odot \mathbf{s}^{X}_e$ 琛ㄧず灏嗙煩闃电殑姣忎竴琛屼箻浠?$\mathbf{s}^{X}_e$ 涓搴旂殑鏍囬噺銆?
        **3. SwiGLU 婵€娲?*

        棣栧厛灏?$\mathbf{H}_e$ 娌垮垪缁村害鎷嗗垎涓?**gate 閮ㄥ垎** $\mathbf{H}_{\text{gate}}$ 鍜?**up 閮ㄥ垎** $\mathbf{H}_{\text{up}}$锛岀劧鍚庡 gate 閮ㄥ垎搴旂敤 SiLU 婵€娲诲嚱鏁帮紝鍐嶄笌 up 閮ㄥ垎閫愬厓绱犵浉涔橈紝寰楀埌涓撳 $e$ 鐨勪腑闂存縺娲昏〃绀?$\mathbf{A}_e$銆?        $$
        \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:,\; :\!\text{intermediateHidden}], \qquad
        \mathbf{H}_{\text{up}} = \mathbf{H}_e[:,\; \text{intermediateHidden}:]
        $$
        $$
        \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
        \quad\in \mathbb{R}^{N_e \times \text{intermediateHidden}}
        $$

        **4. Linear2 鎶曞奖锛堥噺鍖?+ INT4 脳 INT4 鐭╅樀涔?+ 鍙嶉噺鍖栵級**

        Linear2 鎶曞奖浣滀负绗簩灞傜嚎鎬у彉鎹紝灏嗕腑闂存縺娲昏〃绀?$\mathbf{A}_e$ 浠庨珮缁寸┖闂存姇褰卞洖鍘熷鐨勯殣钘忕淮搴?$\text{hidden}$锛屼娇涓撳杈撳嚭鑳藉涓庢畫宸繛鎺ョ瓑鍚庣画鎿嶄綔鍏煎銆?        鍦?A8W4 鍦烘櫙涓嬶紝棣栧厛灏嗘縺娲诲€奸噺鍖栦负 INT8锛屽啀閫氳繃浜岃繘鍒堕噸瑙ｉ噴涓轰袱涓?INT4 鐨勬嫾鎺ヨ鍥撅紝浠ヤ竴娆＄煩闃典箻瀹屾垚璁＄畻锛岃繃绋嬪涓嬨€?
        **(1) 婵€娲婚噺鍖?*

        瀵?$\mathbf{A}_e$ 鐨勬瘡涓€琛岃绠楃缉鏀惧洜瀛愬苟閲忓寲鑷?INT8锛?        $$
        s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
        $$
        寰楀埌涓撳 $e$ 鍦?Linear2 璁＄畻鏃剁殑婵€娲荤缉鏀惧洜瀛?$\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$锛屽苟璁＄畻閲忓寲缁撴灉锛?        $$
        \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}
        $$

        **(2) 婵€娲婚噸瑙ｉ噴涓?INT4**

        灏?INT8 婵€娲诲紶閲?$\mathbf{A}_e^{\text{int8}} \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediateHidden}}$ 浜岃繘鍒堕噸瑙ｉ噴涓轰袱涓?INT4 浜ゆ浛鎷兼帴鐨勮鍥撅細

        $$
        \mathbf{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}} \left( \mathbf{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}
        $$

        鍏朵腑楂樹綅鍜屼綆浣嶅垎閲忕洿鎺ョ敱 $\mathbf{A}_e^{\text{int4}}$ 鐨勫伓鏁拌鍜屽鏁拌缁欏嚭锛?
        $$
        \mathbf{A}_e^{\text{high}} = \mathbf{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}, \quad
        \mathbf{A}_e^{\text{low}} = \mathbf{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{N_e \times \text{intermediateHidden}}
        $$

        瀹冧滑鍦ㄦ暟鍊间笂鍙互鐢卞師 INT8 缁忓涓嬭绠楀緱鍒帮細

        $$
        \mathbf{A}_e^{\text{high}} = \left\lfloor \frac{\mathbf{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
        \mathbf{A}_e^{\text{low}} = (\mathbf{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
        $$

        鎭㈠鍏崇郴涓?$\mathbf{A}_e^{\text{int8}} = 16\mathbf{A}_e^{\text{high}} + (\mathbf{A}_e^{\text{low}} + 8)$銆傜敱浜?$\mathrm{bitcast}$ 浠呮敼鍙樼被鍨嬭鍥撅紝$\mathbf{A}_e^{\text{int4}}$ 涓?$\mathbf{A}_e^{\text{int8}}$ 鍏变韩搴曞眰鐗╃悊鍐呭瓨锛屾棤闇€浠讳綍鏁版嵁閲嶆帓鎴栨嫹璐濄€?
        **(3) INT4 脳 INT4 鐭╅樀涔樹笌鏉冮噸鍙嶉噺鍖?*

        灏嗛噸瑙ｉ噴鍚庣殑 INT4 婵€娲昏鍥?$\mathbf{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2N_e \times \text{intermediateHidden}}$ 涓庢潈閲嶇煩闃?$\mathbf{W}_2[e] \in \mathbb{R}^{\text{intermediateHidden} \times \text{hidden}}$ 鎵ц鐭╅樀涔橈紝骞跺簲鐢ㄦ潈閲嶇缉鏀惧洜瀛?$\mathbf{s}^{W2}_e$ 杩涜鍙嶉噺鍖栵紝寰楀埌缁撴灉锛?
        $$
        \mathbf{D}_e = \bigl( \mathbf{A}_e^{\text{int4}} \cdot \mathbf{W}_2[e] \bigr) \odot \mathbf{s}^{W2}_e
        \;\in\; \mathbb{R}^{2N_e \times \text{hidden}}
        $$

        灏嗙粨鏋滄寜鍋舵暟琛屽拰濂囨暟琛屽垎鍒涓哄涓嬬殑鐭╅樀瑙嗗浘锛屽畠浠嵆瀵瑰簲浜?$\mathbf{A}_e^{\text{high}}$ 鍜?$\mathbf{A}_e^{\text{low}}$ 鐨勮绠楃粨鏋滐細

        $$
        \mathbf{D}_e^{\text{high}} = \mathbf{D}_e[0::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}, \qquad
        \mathbf{D}_e^{\text{low}}  = \mathbf{D}_e[1::2, :] \in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        **(4) 绮惧害琛ュ伩涓庢縺娲诲弽閲忓寲**

        鍒╃敤鍋忕疆 $\mathbf{B}_2[e]$ 鍜屾縺娲荤缉鏀惧洜瀛?$\mathbf{s}^{A_e}_e$锛堣鍚戦噺锛夊垎鍒繘琛岀簿搴﹁ˉ鍋垮拰婵€娲诲弽閲忓寲锛屾渶缁堣緭鍑轰负锛?
        $$
        \mathbf{Y}_e = \Bigl( 16 \cdot \mathbf{D}_e^{\text{high}} + \mathbf{D}_e^{\text{low}} + \mathbf{B}_2[e] \Bigr) \odot \mathbf{s}^{A_e}_e
        \quad\in \mathbb{R}^{N_e \times \text{hidden}}
        $$

        鍏朵腑 $\odot \mathbf{s}^{A_e}_e$ 琛ㄧず灏嗙煩闃电殑姣忎竴琛屼箻浠?$\mathbf{s}^{A_e}_e$ 涓搴旂殑鏍囬噺銆?
        缁忚繃浠ヤ笂璁＄畻锛屼笓瀹?$e$ 鐨勬瘡涓€琛岃緭鍑哄搴斿叾鎵规涓殑涓€涓?token銆傚浜?$\text{token}_i$锛屽畠鍦ㄤ笓瀹?$e_{i,k}$ 涓殑杈撳嚭琛屽嵆涓?$\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$銆?
    - **Token Combine**

        Combine 璐熻矗鏀堕泦鎵€鏈変笓瀹惰绠楀嚭鐨勮緭鍑哄悜閲忥紝鎸夌収姣忎釜 token 鍘熷厛鍒嗛厤鍒扮殑涓撳鏉冮噸杩涜鍔犳潈姹傚拰锛屾渶缁堜负姣忎釜 token 鐢熸垚涓€涓瀺鍚堝悗鐨勮緭鍑恒€傚埄鐢ㄤ箣鍓嶈褰曠殑浣嶇疆绱㈠紩 $\operatorname{row}(i,k)$锛屼粠涓撳 $e_{i,k}$ 鐨勮緭鍑虹煩闃典腑鏀跺洖灞炰簬 $\text{token}_i$ 鐨勮锛屽苟涓庨棬鎺ф潈閲嶇浉涔樺悗姹傚拰锛?        $$
        \mathbf{y}_i = \sum_{k=0}^{\text{topK} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
        \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
        $$
        鍏朵腑 $w_k = \mathbf{G}[i,k]$ 涓?$\text{token}_i$ 瀵逛笓瀹?$e_{i,k}$ 鐨勯棬鎺ф潈閲嶃€?
        鎵€鏈?token 鐨勮緭鍑烘寜杈撳叆椤哄簭鍫嗗彔涓烘渶缁堣緭鍑?$\mathbf{Y} \in \mathbb{R}^{\text{totalNumTokens} \times \text{hidden}}$銆?
    </details>

    <details>
    <summary> A8W8-FP 閲忓寲鍦烘櫙</summary>

    绗竴闃舵瀵硅緭鍏?Token 鎸変笓瀹跺垎缁勬敹闆嗗悗鍋?MXFP8 閲忓寲锛岀敓鎴愬悇涓撳鐨勯噺鍖栬緭鍏ヤ笌缂╂斁鍥犲瓙锛?
    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
    $$

    璇存槑锛氭牴鎹?`topkIds` 灏?Token 鎸変笓瀹舵帓搴忔敹闆嗭紝$\mathcal{T}_e$ 涓哄垎閰嶅埌涓撳 $e$ 鐨?Token 绱㈠紩闆嗗悎锛?E_{local}$琛ㄧず褰撳墠涓撳鏀跺埌鐨勬渶澶oken鏁帮紝姣忎釜涓撳鏁板€煎彲鑳戒笉鍚岋紝$X[\mathcal{T}_e]$ 涓哄搴旂殑瀛愮煩闃点€?\mathrm{Q}_{\text{MX}}$ 琛ㄧず MX 閫愮粍閲忓寲锛坓roup size = 32锛夛紝瀵规瘡缁?32 涓厓绱犳彁鍙栧叡浜寚鏁板悗閲忓寲涓?FP8 鐩爣绫诲瀷锛團LOAT8_E5M2 鎴?FLOAT8_E4M3FN锛夛紝鍚屾椂杈撳嚭 FLOAT8_E8M0 缂╂斁鍥犲瓙銆傞噺鍖栧悗鐨勬暟鎹皢浣滀负 GMM1 鐨勮緭鍏ャ€?
    绗簩闃舵瀵规瘡涓笓瀹舵墽琛?GMM1 鐭╅樀涔樻硶锛堝皢 $W_1$ 娌垮垪鏂瑰悜鍒嗕负涓ゅ崐鍒嗗埆璁＄畻锛夈€丼wiGLU 婵€娲诲拰 MX 閲忓寲锛?
    $$
    Z_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(x)}, S_{1,e}^{(x)}), \quad Z_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(y)}, S_{1,e}^{(y)})
    $$

    $$
    U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
    $$

    $$
    \hat{U}_e,\ S_{U,e} = \mathrm{Q}_{\text{MX}}(U_e)
    $$

    璇存槑锛氬皢 $W_1$ 鐨勫墠 $N/2$ 鍒?$W_{1,e}^{(x)}$ 鍜屽悗 $N/2$ 鍒?$W_{1,e}^{(y)}$ 鍒嗗埆涓?MX 鍙嶉噺鍖栧悗鐨勮緭鍏ュ仛鐭╅樀涔樻硶锛屽緱鍒?Swish 鍒嗘敮 $Z_e^{(x)}$ 鍜岄棬鎺у垎鏀?$Z_e^{(y)}$銆係wiGLU 婵€娲诲涓や釜鍒嗘敮鍋氶€愬厓绱犱箻绉?$x \cdot \sigma(x) \cdot y$锛屽叾涓?$\sigma$ 涓?Sigmoid 鍑芥暟锛屽皢涓棿缁村害浠?$N$ 鍑忓崐涓?$N/2$銆傞殢鍚庡 SwiGLU 杈撳嚭鍋?MX 閲忓寲锛屽緱鍒?GMM2 鐨勯噺鍖栬緭鍏?$\hat{U}_e$銆?
    绗笁闃舵瀵规瘡涓笓瀹舵墽琛?GMM2 鐭╅樀涔樻硶锛屽苟灏嗙粨鏋滄寜鐩爣 Rank 鍒嗗彂锛?
    $$
    O_e = \mathrm{DQ}_{\text{MX}}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
    $$

    璇存槑锛氬皢閲忓寲鍚庣殑 SwiGLU 杈撳嚭涓庣浜岀粍鏉冮噸 $W_2$ 鍋?MX 鍙嶉噺鍖栧悗鐨勭煩闃典箻娉曪紝灏?$N/2$ 缁翠腑闂磋〃绀烘槧灏勫洖 $H$ 缁撮殣钘忕┖闂达紝寰楀埌姣忎釜涓撳鐨勮緭鍑?$O_e$銆傝绠楀畬鎴愬悗閫氳繃 RDMA peermem 灏嗙粨鏋滄寜鐩爣 Rank 鐨勪笓瀹跺亸绉诲湴鍧€鍐欏叆杩滅锛屽疄鐜拌法 Rank 鑱氬悎銆?
    绗洓闃舵瀵规墍鏈?Token 鎸夎矾鐢辨潈閲嶅姞鏉冩眰鍜岋紝鎭㈠涓轰笌杈撳叆鐩稿悓褰㈢姸鐨勮緭鍑猴細

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
    $$

    璇存槑锛氬姣忎釜 Token $i$锛屾牴鎹帓搴忓悗鐨勮矾鐢辩储寮?$\pi(i,k)$ 浠庤仛鍚堝悗鐨勪笓瀹剁粨鏋滀腑鍙栧嚭瀵瑰簲琛岋紝鎸?`topkWeights` 涓殑鏉冮噸閫愬厓绱犲姞鏉冪疮鍔狅紝寰楀埌鏈€缁堣緭鍑?$Y$銆?
    鍏朵腑锛?X$ 琛ㄧず鍙傛暟 `x`锛?W$ 琛ㄧず鍙傛暟 `topkWeights`锛?W_1$ 琛ㄧず鍙傛暟 `weight1`锛?W_2$ 琛ㄧず鍙傛暟 `weight2`锛?Y$ 琛ㄧず鍙傛暟 `y`锛?E_{\text{local}}$ 琛ㄧず `localMoeExpertNum = moeExpertNum / epWorldSize`锛堟瘡涓?Rank 鐨勮矾鐢?MoE 涓撳鏁帮級锛?K$ 琛ㄧず `topkIds` 鐨勭浜岀淮搴︺€?
    灞€閮ㄥ彉閲忚鏄庯細
    - $\mathcal{T}_e$锛氳璺敱鍒颁笓瀹?$e$ 鐨?Token 绱㈠紩闆嗗悎锛岀敱 `topkIds` 鎺掑簭鍚庣‘瀹氥€?    - $\hat{X}_e,\ S_{X,e}$锛氫笓瀹?$e$ 鐨勯噺鍖栬緭鍏ュ強鍏?MX 缂╂斁鍥犲瓙锛岀涓€闃舵涓棿缁撴灉銆?    - $W_{1,e}^{(x)}$銆?W_{1,e}^{(y)}$锛?W_1$ 瀵瑰簲涓撳 $e$ 鐨勫墠 $N/2$ 鍒楀拰鍚?$N/2$ 鍒楀瓙鐭╅樀锛岀敱 `weight1` 鎸?SwiGLU 鎷嗗垎鎺ㄥ銆?    - $S_{1,e}^{(x)}$銆?S_{1,e}^{(y)}$锛?W_{1,e}^{(x)}$ 鍜?$W_{1,e}^{(y)}$ 瀵瑰簲鐨?MX 缂╂斁鍥犲瓙锛屼粠 `weightScales1` 鎸夌淮搴︽埅鍙栥€?    - $S_{2,e}$锛?W_{2,e}$ 瀵瑰簲鐨?MX 缂╂斁鍥犲瓙锛屾潵鑷弬鏁?`weightScales2`銆?    - $Z_e^{(x)},\ Z_e^{(y)}$锛欸MM1 鐨勪袱璺煩闃典箻娉曡緭鍑猴紙Swish 鍒嗘敮鍜岄棬鎺у垎鏀級锛屼腑闂寸粨鏋溿€?    - $U_e$锛歋wiGLU 婵€娲昏緭鍑猴紝缁村害 $m_e \times N/2$锛屼腑闂寸粨鏋溿€?    - $\hat{U}_e,\ S_{U,e}$锛氶噺鍖栧悗鐨?SwiGLU 杈撳嚭鍙婂叾 MX 缂╂斁鍥犲瓙锛屼腑闂寸粨鏋溿€?    - $O_e$锛欸MM2 鐨勪笓瀹剁骇杈撳嚭锛岀淮搴?$m_e \times H$锛屼腑闂寸粨鏋溿€?    - $\pi(i, k)$锛歍oken $i$ 鐨勭 $k$ 涓?top-k 涓撳鍦ㄥ睍寮€鎺掑簭鍚庣殑琛岀储寮曪紝鐢辫矾鐢辨帓搴忕‘瀹氥€?    - $\mathrm{Q}_{\text{MX}}(\cdot)$锛歁X 閫愮粍閲忓寲鎿嶄綔锛宐lock size = 32锛岃緭鍑?FP8 鏁版嵁鍜?E8M0 缂╂斁鍥犲瓙銆?    - $\mathrm{DQ}_{\text{MX}}(\cdot)$锛歁X 閫愮粍鍙嶉噺鍖栨搷浣滐紝鍦?matmul 鍐呴儴闅愬紡鎵ц銆?    </details>

    <details>
    <summary> A8W4-FP 閲忓寲鍦烘櫙</summary>

    绗竴闃舵锛圱oken 閫夋嫨銆侀噺鍖栦笌 Dispatch锛夛細

    瀵规湰 rank 鐨勮緭鍏?Token $X \in \mathbb{R}^{B\times H}$锛屾牴鎹?`topkIds` 寰楀埌姣忎釜涓撳 $e$ 瀵瑰簲鐨?Token 涓嬫爣闆嗗悎 $T_e$锛屽苟灏嗛€変腑鐨?BF16 Token 鎸?32 涓厓绱犱竴缁勯噺鍖栦负 MXFP8 E4M3锛?
    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    鍏朵腑锛?\hat{X}_e$ 鐨勬暟鎹被鍨嬩负 `FLOAT8_E4M3FN`锛?S_{X,e}$ 鐨勬暟鎹被鍨嬩负 `FLOAT8_E8M0`銆傞殢鍚庡皢閲忓寲鍚庣殑 Token 鍙婂叾缂╂斁鍥犲瓙鍙戦€佸埌涓撳鎵€鍦?rank銆?
    璺敱 MoE 涓撳鐨勭涓€灞傚拰绗簩灞傛潈閲?$W_{1,e}$銆?W_{2,e}$ 鍧囦负 MXFP4 E2M1 鏁版嵁锛岀缉鏀惧洜瀛愪负 E8M0锛?e$ 鐨勮寖鍥翠负 $[0,\text{localMoeExpertNum})$銆傚叡浜笓瀹舵潈閲嶅崟鐙敱 `sharedWeight1`銆乣sharedWeight2` 鎻愪緵锛屽叾绗竴缁翠负 `sharedExpertNumPerRank`銆侫8W4 kernel 鍦ㄧ煩闃典箻 Prologue 涓鐞?FP4 鏉冮噸锛岄€佸叆鐭╅樀涔樼殑閫昏緫鏁版嵁娴佷负 FP8 婵€娲讳箻 FP4 鏉冮噸銆?
    绗簩闃舵锛圙MM1銆丼wiGLU 涓庡啀娆￠噺鍖栵級锛?
    瀵逛笓瀹?$e$ 鏀跺埌鐨?Token锛岀涓€灞傚垎缁勭煩闃典箻鍜?SwiGLU 璁＄畻涓猴細

    $$
    Z_x = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(x)},S_{1,e}^{(x)}\right),
    $$

    $$
    Z_y = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(y)},S_{1,e}^{(y)}\right),
    $$

    $$
    U_e = Z_x \odot \sigma(Z_x) \odot Z_y.
    $$

    SwiGLU 杈撳嚭缁х画鎸?32 涓厓绱犱竴缁勯噺鍖栦负 MXFP8 E4M3锛屼緵绗簩灞傜煩闃典箻浣跨敤锛?
    $$
    \hat{U}_e,\;S_{U,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(U_e\right).
    $$

    绗笁闃舵锛圙MM2锛夛細

    绗簩灞傜煩闃典箻浠嶄负 A8W4锛屽嵆 FP8 E4M3 婵€娲讳箻 FP4 E2M1 鏉冮噸锛?
    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{U}_e,S_{U,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    绗洓闃舵锛圕ombine 涓庡姞鏉冨悎骞讹級锛?
    灏嗗悇涓撳杈撳嚭閫佸洖鍘?rank锛屽苟鏍规嵁 `topkWeights` 鍋氬姞鏉冨悎骞讹細

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)],
    $$

    鏈€缁堣緭鍑?$Y$ 鐨勬暟鎹被鍨嬩负 BF16銆侫8W4-FP 鐨勪富瑕佹暟鎹被鍨嬫祦涓猴細`BF16 -> MXFP8 E4M3 -> A8W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`銆?    </details>

    <details>
    <summary> A4W4-FP 閲忓寲鍦烘櫙</summary>

    绗竴闃舵锛圱oken 閫夋嫨銆侀噺鍖栦笌 Dispatch锛夛細

    瀵规瘡涓笓瀹?$e$ 瀵瑰簲鐨?Token 闆嗗悎 $T_e$锛屽皢閫変腑鐨?BF16 Token 鎸?32 涓厓绱犱竴缁勯噺鍖栦负 MXFP4 E2M1锛?
    $$
    \hat{X}_e,\;S_{X,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(X[T_e]\right),
    $$

    鍏朵腑锛?\hat{X}_e$ 鐨勬暟鎹被鍨嬩负 `FLOAT4_E2M1`锛?S_{X,e}$ 鐨勬暟鎹被鍨嬩负 `FLOAT8_E8M0`銆傝矾鐢?MoE 涓撳鐨勭涓€灞傚拰绗簩灞傛潈閲?$W_{1,e}$銆?W_{2,e}$ 鍧囦负 MXFP4 E2M1锛?e$ 鐨勮寖鍥翠负 $[0,\text{localMoeExpertNum})$锛涘叡浜笓瀹舵潈閲嶇殑绗竴缁翠负 `sharedExpertNumPerRank`銆傛潈閲嶇缉鏀惧洜瀛愪负 E8M0銆?
    绗簩闃舵锛圓4W4 GMM1銆丼wiGLU 涓庤緭鍑虹被鍨嬫彁鍗囷級锛?
    绗竴灞傚垎缁勭煩闃典箻涓?A4W4锛?
    $$
    Z_x = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(x)},S_{1,e}^{(x)}\right),
    $$

    $$
    Z_y = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{X}_e,S_{X,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{1,e}^{(y)},S_{1,e}^{(y)}\right),
    $$

    $$
    U_e = Z_x \odot \sigma(Z_x) \odot Z_y.
    $$

    杩欓噷涓嶈兘缁х画鎶?SwiGLU 杈撳嚭閲忓寲涓?FP4銆俴ernel 鍦?`QuantMode == E2M1_QUANT` 鏃讹紝灏?`SwigluQuantOutType` 鎸囧畾涓?`fp8_e4m3fn_t`锛屽洜姝よ緭鍑轰細鎻愬崌涓?MXFP8 E4M3锛?
    $$
    \hat{U}_e,\;S_{U,e}
    = \mathrm{Q}_{\mathrm{MX}}\!\left(U_e\right).
    $$

    绗笁闃舵锛圓8W4 GMM2锛夛細

    鐢变簬 SwiGLU 閲忓寲杈撳嚭涓?FP8 E4M3锛岃€岀浜屽眰鏉冮噸浠嶄负 FP4 E2M1锛屽洜姝ょ浜屽眰鐭╅樀涔樺疄闄呬负 A8W4锛岃€屼笉鏄?A4W4锛?
    $$
    O_e = \mathrm{DQ}_{\mathrm{MX}}\!\left(\hat{U}_e,S_{U,e}\right)
          \cdot \mathrm{DQ}_{\mathrm{MX}}\!\left(W_{2,e},S_{2,e}\right).
    $$

    绗洓闃舵锛圕ombine 涓庡姞鏉冨悎骞讹級锛?
    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,k] \cdot O[\pi(i,k)],
    $$

    鏈€缁堣緭鍑?$Y$ 鐨勬暟鎹被鍨嬩负 BF16銆侫4W4-FP 鐨勫畬鏁存暟鎹被鍨嬫祦涓猴細`BF16 -> MXFP4 E2M1 -> A4W4 GMM1 -> MXFP8 E4M3 -> A8W4 GMM2 -> BF16`銆傚叾涓墍鏈?MX 缂╂斁鍥犲瓙鐨勭被鍨嬪潎涓?`FLOAT8_E8M0`锛岄噺鍖栫矑搴﹀潎涓?32 涓繛缁厓绱犮€?    </details>

## 鍙傛暟璇存槑

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>鍙傛暟鍚?/th>
   <th>杈撳叆/杈撳嚭/灞炴€?/th>
   <th>鎻忚堪</th>
   <th>鏁版嵁绫诲瀷</th>
   <th>鏁版嵁鏍煎紡</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>杈撳叆</td>
   <td>鏈崱閫氫俊鍩熶俊鎭暟鎹€?/td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>杈撳叆</td>
   <td>MoE灞傝緭鍏ョ殑token闅愯棌鐘舵€併€?/td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkIds</td>
   <td>杈撳叆</td>
   <td>涓撳绱㈠紩鐭╅樀锛岃〃绀烘瘡涓猼oken閫夋嫨鐨則opK涓笓瀹躲€傚厓绱犲彇鍊艰寖鍥翠负[0, moeExpertNum)锛屼笖鍚屼竴token閫夋嫨鐨則opK涓笓瀹朵笉鑳介噸澶嶃€?/td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topkWeights</td>
   <td>杈撳叆</td>
   <td>琛ㄧずMoE妯″瀷鐨勪笓瀹堕棬鎺х綉缁滀负褰撳墠杈撳叆Token閫夊嚭鐨則opK涓笓瀹舵墍瀵瑰簲鐨勯棬鎺ф潈閲嶇郴鏁般€?/td>
   <td>FP32銆丅F16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td>杈撳叆</td>
   <td>MoE涓撳缃戠粶绗竴绾挎€у眰鐨勬潈閲嶇煩闃碉紙鍖呮嫭闂ㄦ帶涓庝笂鎶曞奖锛夛紝鐢ㄤ簬灏嗚緭鍏ユ槧灏勮嚦涓棿缁村害锛岃緭鍑轰緵缁欐縺娲诲嚱鏁般€傝矾鐢?MoE 涓撳鏁颁负 <code>localMoeExpertNum</code>銆?/td>
   <td>BF16銆両NT8銆両NT4銆丗LOAT8_E5M2銆丗LOAT8_E4M3FN銆丗LOAT4_E2M1</td>
   <td>ND銆丗RACTAL_NZ銆丗ORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>杈撳叆</td>
   <td>MoE涓撳缃戠粶绗簩绾挎€у眰鐨勬潈閲嶇煩闃碉紝璐熻矗灏嗘縺娲诲悗鐨勪腑闂寸壒寰佹姇褰卞洖闅愯棌缁村害銆傛暟鎹被鍨嬩笌weight1涓€鑷淬€傝矾鐢?MoE 涓撳鏁颁负 <code>localMoeExpertNum</code>銆?/td>
   <td>BF16銆両NT8銆両NT4銆丗LOAT8_E5M2銆丗LOAT8_E4M3FN銆丗LOAT4_E2M1</td>
   <td>ND銆丗RACTAL_NZ銆丗ORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>weightScales1</td>
   <td>鍙€夎緭鍏?/td>
   <td>MoE涓撳缃戠粶绗竴绾挎€у眰鐨勬潈閲嶇煩闃电殑閲忓寲缂╂斁鍥犲瓙銆?/td>
   <td>FLOAT8_E8M0銆乁INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScales2</td>
   <td>鍙€夎緭鍏?/td>
   <td>MoE涓撳缃戠粶绗簩绾挎€у眰鐨勬潈閲嶇煩闃电殑閲忓寲缂╂斁鍥犲瓙銆?/td>
   <td>FLOAT8_E8M0銆乁INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias1</td>
   <td>鍙€夎緭鍏?/td>
   <td>MoE涓撳缃戠粶绗竴绾挎€у眰鐨勫亸缃紝浠呬簬A8W4-INT閲忓寲鍦烘櫙涓嬮渶瑕佽鍙傛暟锛岀敤浜庣簿搴﹁ˉ鍋裤€?/td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias2</td>
   <td>鍙€夎緭鍏?/td>
   <td>MoE涓撳缃戠粶绗簩绾挎€у眰鐨勫亸缃紝浠呬簬A8W4-INT閲忓寲鍦烘櫙涓嬮渶瑕佽鍙傛暟锛岀敤浜庣簿搴﹁ˉ鍋裤€?/td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td>鍙€夎緭鍏?/td>
   <td>琛ㄧずtoken鏄惁鍙備笌閫氫俊銆?/td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
    <tr>
   <td>sharedWeight1</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗竴绾挎€у眰鐨勬潈閲嶇煩闃碉紙鍖呮嫭闂ㄦ帶涓庝笂鎶曞奖锛夛紝鐢ㄤ簬灏嗚緭鍏ユ槧灏勮嚦涓棿缁村害锛岃緭鍑轰緵缁欐縺娲诲嚱鏁般€?/td>
   <td>FLOAT8_E5M2銆丗LOAT8_E4M3FN銆丗LOAT4_E2M1</td>
   <td>ND銆丗RACTAL_NZ銆丗ORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeight2</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗簩绾挎€у眰鐨勬潈閲嶇煩闃碉紝璐熻矗灏嗘縺娲诲悗鐨勪腑闂寸壒寰佹姇褰卞洖闅愯棌缁村害銆傛暟鎹被鍨嬩笌weight1涓€鑷淬€?/td>
   <td>FLOAT8_E5M2銆丗LOAT8_E4M3FN銆丗LOAT4_E2M1</td>
   <td>ND銆丗RACTAL_NZ銆丗ORMAT_FRACTAL_NZ_C0_32</td>
  </tr>
  <tr>
   <td>sharedWeightScales1</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗竴绾挎€у眰鐨勬潈閲嶇煩闃电殑閲忓寲缂╂斁鍥犲瓙銆?/td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedWeightScales2</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗簩绾挎€у眰鐨勬潈閲嶇煩闃电殑閲忓寲缂╂斁鍥犲瓙銆?/td>
   <td>FLOAT8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias1</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗竴绾挎€у眰鐨勫亸缃紝鏆備笉鏀寔銆?/td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedBias2</td>
   <td>鍙€夎緭鍏?/td>
   <td>鍏变韩涓撳缃戠粶绗簩绾挎€у眰鐨勫亸缃紝鏆備笉鏀寔銆?/td>
   <td>FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>灞炴€?/td>
   <td>MoE妯″瀷鐨勬€讳笓瀹舵暟閲忋€?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>灞炴€?/td>
   <td>涓撳骞惰閫氫俊鍩熷ぇ灏忋€?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>cclBufferSize</td>
   <td>灞炴€?/td>
   <td>CCL閫氫俊缂撳啿鍖哄ぇ灏忋€?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>maxRecvTokenNum</td>
   <td>鍙€夊睘鎬?/td>
   <td>姣忎釜Rank鏈€澶у彲鎺ユ敹Token鏁般€傞粯璁ゅ€间负0銆傝鍊间负0鏃讹紝浼氭寜鏈€澶у€糱s*epWorldSize*min(topK,localMoeExpertNum)棰勭暀鍐呭瓨澶у皬锛涗笉涓?鏃讹紝灏嗘寜鐓ц緭鍏ュ€煎ぇ灏忛鐣欏唴瀛橈紝璇ュ満鏅笅闇€鐢ㄦ埛淇濊瘉濉叆鍊煎ぇ浜庣瓑浜庢瘡涓猺ank鏈€澶у彲鎺ユ敹鐨凾oken鏁般€?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantMode</td>
   <td>鍙€夊睘鎬?/td>
   <td>dispatch閫氫俊鏃堕噺鍖栨ā寮忋€?琛ㄧず闈為噺鍖栵紙A16W16鍦烘櫙锛夛紝2琛ㄧずINT8閲忓寲锛圓8W8-INT銆丄8W4-INT鍦烘櫙锛夛紝4琛ㄧずMXFP閲忓寲锛圓8W8-FP銆丄8W4-FP銆丄4W4-FP鍦烘櫙锛夈€傞粯璁ゅ€间负0銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatchQuantOutDtype</td>
   <td>鍙€夊睘鎬?/td>
   <td>dispatch閲忓寲鍚庤緭鍑虹殑鏁版嵁绫诲瀷銆傛敮鎸?锛圛NT8锛夈€?3锛團LOAT8_E5M2锛夈€?4锛團LOAT8_E4M3FN锛夈€?96锛團LOAT4_E2M1锛夈€傞粯璁ゅ€间负DT_UNDEFINED銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combineQuantMode</td>
   <td>鍙€夊睘鎬?/td>
   <td>combine閫氫俊鏃剁殑閲忓寲妯″紡銆?琛ㄧず闈為噺鍖栵紝3琛ㄧずMXFP float8_e5m2绫诲瀷锛?琛ㄧずMXFP float8_e4m3绫诲瀷, 榛樿鍊间负0銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>鍙€夊睘鎬?/td>
   <td>棰勭暀鍙傛暟锛屾殏涓嶆敮鎸併€傞粯璁ゅ€间负""銆?/td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>numMaxTokensPerRank</td>
   <td>鍙€夊睘鎬?/td>
   <td>姣忓紶鍗′笂鐨則oken鏁伴噺銆傚綋姣忎釜rank鐨刵umTokens涓嶅悓鏃讹紝涓烘渶澶х殑numTokens澶у皬銆傞粯璁ゅ€间负0銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>activation</td>
   <td>鍙€夊睘鎬?/td>
   <td>婵€娲诲嚱鏁扮被鍨嬨€傞粯璁ゅ€间负"swiglu"銆傚綋鍓嶄粎鏀寔"swiglu"銆?/td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>activationClamp</td>
   <td>鍙€夊睘鎬?/td>
   <td>婵€娲诲嚱鏁拌緭鍏ョ殑瀵圭О鎴柇闃堝€硷紝灏嗚緭鍏ュ紶閲忛檺鍒跺湪[-activationClamp, activationClamp]鍖洪棿鍐呫€傞粯璁ゅ€间负float鏈€澶у€硷紝琛ㄧず涓嶆埅鏂紝鍊奸渶鈮?銆?/td>
   <td>FLOAT</td>
   <td></td>
  </tr>
  <tr>
   <td>activationOutDtype</td>
   <td>鍙€夊睘鎬?/td>
   <td>婵€娲诲嚱鏁拌緭鍑虹殑鏁版嵁绫诲瀷銆傞粯璁ゅ€间负DT_UNDEFINED銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight1</td>
   <td>鍙€夊睘鎬?/td>
   <td>weight1鏄惁杞疆銆傞粯璁ゅ€间负false銆?/td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>transposeWeight2</td>
   <td>鍙€夊睘鎬?/td>
   <td>weight2鏄惁杞疆銆傞粯璁ゅ€间负false銆?/td>
   <td>BOOL</td>
   <td></td>
  </tr>
  <tr>
   <td>weight1Interleave</td>
   <td>鍙€夊睘鎬?/td>
   <td>weight1浜ら敊鍙傛暟銆傞鐣欏弬鏁帮紝榛樿鍊间负0銆?/td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>y</td>
   <td>杈撳嚭</td>
   <td>璁＄畻杈撳嚭缁撴灉锛屾暟鎹被鍨嬩笌杈撳叆x鐩稿悓銆?/td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNums</td>
   <td>杈撳嚭</td>
   <td>鏈崱姣忎釜涓撳瀹為檯鏀跺埌鐨則oken鏁伴噺銆?/td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 绾︽潫璇存槑

- **棰勭暀鍜岄潪瀵瑰鍙傛暟璇存槑**锛?  - 鍙傛暟琛ㄦ牸涓殑閮ㄥ垎鍙傛暟銆侀儴鍒嗘暟鎹被鍨嬫殏鏈澶栨彁渚涳紝涓洪鐣欐垨鍐呴儴瀹炵幇浣跨敤銆傛帴鍙ｅ弬鏁扮殑浠嬬粛鍙婂叾绾︽潫鍦ㄦ帴鍙ｆ枃妗MegaMoE绠楀瓙鎺ュ彛鏂囨。](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)涓缁嗚鏄庛€?
- **鍙傛暟涓€鑷存€х害鏉?*锛?  - 璋冪敤绠楀瓙杩囩▼涓娇鐢ㄧ殑`moeExpertNum`銆乣maxRecvTokenNum`銆乣dispatchQuantMode`銆乣dispatchQuantOutDtype`銆乣numMaxTokensPerRank`绛夊弬鏁板彇鍊硷紝鎵€鏈夊崱闇€淇濇寔涓€鑷达紝缃戠粶涓笉鍚屽眰涓篃闇€淇濇寔涓€鑷淬€?
- **閫氫俊鍩熷拰缁勭綉绾︽潫**锛?  - 鎵€鏈夊崱鐨刞epWorldSize`銆乣cclBufferSize`鍙傛暟鍙栧€奸渶淇濇寔涓€鑷淬€?  - 閫氫俊鍩熷悇鑺傜偣鐨勯┍鍔ㄧ増鏈簲褰撶浉鍚屻€?  - <term>Atlas A2 璁粌绯诲垪浜у搧/Atlas A2 鎺ㄧ悊绯诲垪浜у搧</term>锛氬鏈洪€氫俊鍩熻姹備氦鎹㈡満缁勭綉锛屼笉鏀寔鍙屾満鐩磋繛缁勭綉銆?  - <term>Atlas A3 璁粌绯诲垪浜у搧/Atlas A3 鎺ㄧ悊绯诲垪浜у搧</term>锛氬鏈洪€氫俊鍩熻姹傚湪涓€涓秴鑺傜偣鍐咃紝涓嶆敮鎸佸弻鏈虹洿杩炵粍缃戝拰璺ㄨ秴鑺傜偣缁勭綉銆?  - <term>Ascend 950PR/Ascend 950DT</term>锛氫粎鏀寔UB Memory閫氫俊鍗忚銆?- **鍙傛暟绾︽潫**锛?  - **<term>Atlas A2 璁粌绯诲垪浜у搧/Atlas A2 鎺ㄧ悊绯诲垪浜у搧</term>銆?term>Atlas A3 璁粌绯诲垪浜у搧/Atlas A3 鎺ㄧ悊绯诲垪浜у搧</term>**锛?    - 鍦烘櫙閰嶅鐭╅樀锛?
      | 鍦烘櫙 | x | weight1 | weight2 | weightScales1 | weightScales2 | bias1 | bias2 | y | dispatchQuantMode | dispatchQuantOutDtype |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | A16W16 | BF16 | BF16 | BF16 | 鈥?| 鈥?| 鈥?| 鈥?| BF16 | 0 | 鈥?|
      | A8W8-INT | BF16 | INT8 | INT8 | UINT64 | UINT64 | 鈥?| 鈥?| BF16 | 2 | 1锛圛NT8锛?|
      | A8W4-INT | BF16 | INT4(INT32) | INT4(INT32) | UINT64 | UINT64 | FP32 | FP32 | BF16 | 2 | 1锛圛NT8锛?|

  - **<term>Ascend 950PR/Ascend 950DT</term>**锛?    - BS锛坸.dim0锛夋敮鎸乕1, maxBS], maxBS = (8 * (totalUbSize - 48K)) / (topK * (64 + epWorldSize)), 鍏朵腑totalUbSize涓篣B鎬诲ぇ灏忥紝950绯诲垪浜у搧璇ュ€兼敮鎸佸埌褰撳墠鐜鑳界敵璇风殑鏈€澶у唴瀛樸€?    - H锛坸.dim1锛夋敮鎸?024銆?048銆?072銆?096銆?120銆?144銆?168銆?192銆?    - topK锛坱opkIds.dim1锛夋敮鎸乕1, 32]銆?    - expertPerRank 鑼冨洿 [1, 1024]銆?    - hiddenDim锛坵eight1.dim1锛変粎鏀寔1024銆?048銆?072銆?096銆?168銆?    - epWorldSize鑼冨洿 [2, 1024]銆?    - moeExpertNum鑼冨洿 [epWorldSize, 2048]锛屼笖moeExpertNum % epWorldSize == 0銆?    - maxRecvTokenNum鑼冨洿 [0, BS 脳 epWorldSize 脳 min(topK, localMoeExpertNum)]銆?    - dispatchQuantOutDtype浠呮敮鎸?3锛團LOAT8_E5M2锛夋垨24锛團LOAT8_E4M3FN锛夋垨296锛團LOAT4_E2M1锛夈€?    - 褰撳墠鐗堟湰浠呮敮鎸丮XFP閲忓寲妯″紡锛坉ispatchQuantMode = 4锛夛紝dispatch闃舵浣跨敤MX閫愮粍閲忓寲锛坓roup size = 32锛夛紝閲忓寲缂╂斁鍥犲瓙绫诲瀷涓篎LOAT8_E8M0銆?    - combineQuantMode鍙栧€间负0銆?銆?锛?琛ㄧず闈為噺鍖栵紝3琛ㄧずMXFP float8_e5m2绫诲瀷锛?琛ㄧずMXFP float8_e4m3绫诲瀷
    - commAlg蹇呴』涓虹┖瀛楃涓?"銆?    - y鐨勬暟鎹被鍨嬩笌x鐩稿悓銆?    - weight1鐨刣im1锛坔iddenDim锛夊繀椤荤瓑浜巜eight2鐨刣im2鐨勪簩鍊嶏紝杩欐槸鍥犱负SwiGLU婵€娲婚渶瑕佸皢涓棿缁村害浠巋iddenDim鍑忓崐涓篽iddenDim/2銆?    - localMoeExpertNum = moeExpertNum / epWorldSize锛泂haredExpertNumPerRank = sharedWeight1.dim0锛堟湭鍚敤鍏变韩涓撳鏃朵负0锛夛紱expertPerRank = sharedExpertNumPerRank + localMoeExpertNum銆?
  - **MXFP閲忓寲鍦烘櫙绾︽潫**锛?      - weight1 shape涓?localMoeExpertNum, hiddenDim, H)锛寃eight2 shape涓?localMoeExpertNum, H, hiddenDim/2)銆?      - weightScales1 shape涓?localMoeExpertNum, hiddenDim, CeilDiv(H, 64), 2)銆?      - weightScales2 shape涓?localMoeExpertNum, H, CeilDiv(hiddenDim/2, 64), 2)銆?      - sharedWeight1 shape涓?sharedExpertNumPerRank, hiddenDim, H)锛宻haredWeight2 shape涓?sharedExpertNumPerRank, H, hiddenDim/2)銆?      - sharedWeightScales1 shape涓?sharedExpertNumPerRank, hiddenDim, CeilDiv(H, 64), 2)锛宻haredWeightScales2 shape涓?sharedExpertNumPerRank, H, CeilDiv(hiddenDim/2, 64), 2)銆?      - weightScales1鐨刣im3鍜寃eightScales2鐨刣im3蹇呴』绛変簬2銆?      - A8W4-FP鍦烘櫙涓嬶紝FLOAT4_E2M1绫诲瀷鐨剋eight1蹇呴』浣跨敤FORMAT_FRACTAL_NZ_C0_32鏍煎紡銆?      - MXFP鍦烘櫙涓嬶紝dispatchQuantOutDtype=23鏃秝eight1鍜寃eight2蹇呴』涓篎LOAT8_E5M2锛宒ispatchQuantOutDtype=24鏃跺繀椤讳负FLOAT8_E4M3FN锛宒ispatchQuantOutDtype=296鏃跺繀椤讳负FLOAT4_E2M1銆?
## 璋冪敤璇存槑

| 璋冪敤鏂瑰紡  | 鏍蜂緥浠ｇ爜                                  | 璇存槑                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch鎺ュ彛璋冪敤 | - | 閫氳繃[mega_moe](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)PyTorch鎺ュ彛鏂瑰紡璋冪敤mega_moe绠楀瓙銆?|
