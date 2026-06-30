winIn：
A3 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(int32_t)  // tokenPerExpert空间
        +   bs * topK * h * 2       // D 空间
        +   !isQuant ? bs * topK * h * 2 : bs * topK * (h + ALIGN_512)  // A空间
        +   bs * topK * sizeof(float)  // 外带临时计算scale空间
        +   EP * ALIGN_512   // CrossRankSync使用的flag
物理排序：10MB保留区 -> A -> perTokenScale临时（仅量化） -> D -> tokenPerExpert空间 -> CrossRankSync使用的flag

A2 ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(int32_t)  // tokenPerExpert空间
        +   bs * topK * h * 2       // D 空间
        +   !isQuant ? maxRecvTokenNum * h * 2 : maxRecvTokenNum * (h + ALIGN_512)  // A空间
        +   maxRecvTokenNum * sizeof(float)  // scale空间
        +   (EP * 16 + 1) * 4   // CrossRankSync使用的flag
        +   EP * E * 64  // Dispatch Flag
        +   EP * 64     // allgather flag（仅BF16）
物理排序（INT8/W4A8）：10MB保留区 -> A -> perTokenScale -> D -> dispatch -> tokenPerExpert空间 -> CrossRankSync使用的flag
物理排序（BF16）：10MB保留区 -> A -> D -> allgather -> dispatch -> tokenPerExpert空间 -> CrossRankSync使用的flag

winOut（仅A2）：
ccl buff: offsetTokenPerExpert = EP * Align(globalExpertNum, 128) * sizeof(int32_t)  // tokenPerExpert空间
        +   maxRecvTokenNum * h * 2       // D 空间
        +   !isQuant ? bs * topK * h * 2 : bs * topK * (h + ALIGN_512)  // A空间
        +   maxRecvTokenNum * sizeof(float)  // scale空间
        +   (EP * 16 + 1) * 4   // CrossRankSync使用的flag
        +   EP * E * 64  // Dispatch Flag
        +   EP * 64     // allgather flag（仅BF16）
物理排序（INT8/W4A8）：10MB保留区 -> A -> perTokenScale -> D -> dispatch -> tokenPerExpert空间 -> CrossRankSync使用的flag
物理排序（BF16）：10MB保留区 -> A -> D -> allgather -> dispatch -> tokenPerExpert空间 -> CrossRankSync使用的flag