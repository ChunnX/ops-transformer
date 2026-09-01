file(REMOVE_RECURSE
  "_CPack_Packages"
  "libcust_opmaster_rt2.0.pdb"
  "libcust_opmaster_rt2.0.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/cust_opmaster.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
