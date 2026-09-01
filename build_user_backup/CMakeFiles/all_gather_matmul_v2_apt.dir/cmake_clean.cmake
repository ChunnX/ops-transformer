file(REMOVE_RECURSE
  "_CPack_Packages"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/all_gather_matmul_v2_apt.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
