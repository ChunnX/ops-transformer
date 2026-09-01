file(REMOVE_RECURSE
  "../../../libes_math.pdb"
  "../../../libes_math.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/es_math_so.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
