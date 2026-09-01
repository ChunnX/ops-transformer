file(REMOVE_RECURSE
  "libes_math.a"
  "libes_math.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/es_math_a.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
