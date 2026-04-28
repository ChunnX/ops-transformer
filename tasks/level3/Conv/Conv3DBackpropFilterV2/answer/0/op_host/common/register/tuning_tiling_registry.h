#pragma once
#include <cstdint>

#define BEGIN_TUNING_TILING_DATA_DEF(StructName) struct StructName {
#define TUNING_TILING_DATA_FIELD_DEF(Type, FieldName) Type FieldName;
#define END_TUNING_TILING_DATA_DEF };

#define END_TUNING_TILING_DEF };

#define DECLARE_SCHEMA(StructName, ...)
#define FIELD(StructName, FieldName)

namespace tuningtiling {
inline void RegisterTuningTiling(const char* opType) {
    (void)opType;
}

template<typename T>
inline void DECLARE_STRUCT_RELATE_WITH_OP_V2(const char* opType, const T& args) {
    (void)opType;
    (void)args;
}
}

#define DECLARE_STRUCT_RELATE_WITH_OP_V2(OpName, ArgsStruct, ...) \
    namespace tuningtiling { \
        inline void _declare_##OpName() { (void)0; } \
    }
