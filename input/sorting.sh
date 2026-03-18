#!/bin/bash

ROOT_DIR="$1"

if [ -z "$ROOT_DIR" ]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

cd "$ROOT_DIR" || exit 1

find . -type f -name "WALTzER_*_NIR_*" | while read -r file; do

  basename=$(basename "$file")

  # only keep frame 00000 or 00001
  if [[ ! "$basename" =~ _0000[01] ]]; then
    continue
  fi

  # extract target name
  target=$(echo "$basename" | sed -E 's/^WALTzER_([^_]+)_NIR_.*/\1/')

  dest_dir="$ROOT_DIR/$target"
  mkdir -p "$dest_dir"

  cp "$file" "$dest_dir/"

  echo "Copied $basename -> $dest_dir"

done