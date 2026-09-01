file(REMOVE_RECURSE
  "_CPack_Packages"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ops_transformer_kernel.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
