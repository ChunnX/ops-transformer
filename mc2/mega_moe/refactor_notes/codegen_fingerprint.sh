#!/bin/bash
# 带 kernel 身份的 codegen 指纹：输出 "<kernel sha 前缀> <text-sha256> <rodata-sha256>"
set -e
RUN=$1; OUT=$2
W=$(mktemp -d)
cp "$RUN" "$W/pkg.run"; chmod +x "$W/pkg.run"
( cd "$W" && ./pkg.run --noexec --extract=. >/dev/null 2>&1 || ./pkg.run --extract=. >/dev/null 2>&1 )
: > "$OUT"
for o in $(find "$W" -name "MegaMoe_*.o" | sort); do
  sig=""
  for sec in .text .rodata; do
    read -r off sz < <(readelf -S -W "$o" | awk -v s="$sec" '$3==s {print $6, $7; exit}')
    if [ -n "$off" ] && [ "$sz" != "0000000000000000" ]; then
      h=$(dd if="$o" bs=65536 iflag=skip_bytes,count_bytes skip=$((16#$off)) count=$((16#$sz)) status=none | sha256sum | awk '{print $1}')
    else
      h="empty"
    fi
    sig="$sig$h "
  done
  echo "$(basename "$o" .o) $sig"
done | sort >> "$OUT"
rm -rf "$W"
echo "kernels: $(wc -l < "$OUT")  unique sigs: $(awk '{$1="";print}' "$OUT" | sort -u | wc -l)  -> $OUT"
