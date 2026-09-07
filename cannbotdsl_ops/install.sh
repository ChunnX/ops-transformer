#!/bin/sh
set -eu
prefix=${1:?Usage: flash-kda.run -- /absolute/install/prefix}
case "$prefix" in /*) ;; *) echo 'Install prefix must be absolute' >&2; exit 2;; esac
if [ -e "$prefix/packages" ]; then
    echo 'Install prefix already contains packages; choose a fresh prefix' >&2
    exit 2
fi
mkdir -p "$prefix"
cp -a packages "$prefix/"
printf 'Installed FlashKda frozen operators under %s/packages\n' "$prefix"
