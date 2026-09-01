file(REMOVE_RECURSE
  "_CPack_Packages"
  "CMakeFiles/modify_vendor"
  "scripts/install.sh"
  "scripts/upgrade.sh"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/modify_vendor.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
