cd visor_data/Interpolations-DenseAnnotations/train
find . -type f -name '*.zip' -execdir sh -c 'unzip -o "$1" && rm -f "$1"' sh '{}' \;