A3 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(float)  // cumsum空间
        +   bs * topK * h * 2       // D 空间
        +   !isQuant ? bs * topK * h * 2 : bs * topK * (h + ALIGN_512)  // A空间
        +   bs * topK * sizeof(float)  // 外带临时计算scale空间
        +   EP * ALIGN_512   // CrossRankSync使用的flag
物理排序：A -> perTokenScale临时（仅量化） -> D -> cumsum空间 -> CrossRankSync使用的flag

A2 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(float)  // cumsum空间
        +   bs * topK * h * 2       // D 空间
        +   !isQuant ? bs * topK * h * 2 : bs * topK * (h + ALIGN_512)  // A空间
        +   maxRecvTokenNum * sizeof(float)  // scale空间
        +   (EP * 16 + 1) * 4   // CrossRankSync使用的flag
        +   EP * E * 64  // Dispatch Flag
        +   EP * 64     // allgather flag
物理排序：A -> perTokenScale（单独发送，仅量化） -> D -> allgather空间（onlybf16） -> dispatch空间 -> cumsum空间 -> CrossRankSync使用的flag