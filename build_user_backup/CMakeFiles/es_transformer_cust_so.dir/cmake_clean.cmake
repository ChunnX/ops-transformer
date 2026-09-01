file(REMOVE_RECURSE
  "_CPack_Packages"
  "libes_transformer_cust.pdb"
  "libes_transformer_cust.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/es_transformer_cust_so.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
