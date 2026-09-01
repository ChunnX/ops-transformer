file(REMOVE_RECURSE
  "_CPack_Packages"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/all_gather_matmul_v3_apt_ascend950.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
