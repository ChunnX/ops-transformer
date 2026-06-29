A3 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(float)  // cumsum空间
        +   bs * topK * h * 2       // D 空间
        +   isQuant ? maxRecvTokenNum * h * 2 : maxRecvTokenNum * (h + ALIGN_512)  // A空间
        +   maxRecvTokenNum * sizeof(float)  // 外带临时计算scale空间
        +   EP * ALIGN_512   // CrossRankSync使用的flag
物理排序：A -> perTokenScale临时 -> D -> cumsum空间 -> CrossRankSync使用的flag

A2 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(float)  // cumsum空间
        +   bs * topK * h * 2       // D 空间
        +   isQuant ? maxRecvTokenNum * h * 2 : maxRecvTokenNum * h  // A空间
        +   maxRecvTokenNum * sizeof(float)  // scale空间
        +   EP * ALIGN_512   // CrossRankSync使用的flag
        +   EP * E * 64  // Dispatch Flag
物理排序：A -> perTokenScale临时 -> D -> dispatch空间 -> cumsum空间 -> CrossRankSync使用的flag