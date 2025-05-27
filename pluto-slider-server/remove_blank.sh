#!/bin/sh
# PlutoPDF.jl creates blank pages when using the presentation mode
# This script removes them
# See https://superuser.com/questions/1248528/remove-blank-pages-from-pdf-from-command-line
IN="$1"
OUT="$2"
PAGES=$(pdfinfo "$IN" | grep ^Pages: | tr -dc '0-9')

non_blank() {
    for i in $(seq 1 $PAGES)
    do
        PERCENT=$(gs -o -  -dFirstPage=${i} -dLastPage=${i} -sDEVICE=inkcov "$IN" | grep CMYK | nawk 'BEGIN { sum=0; } {sum += $1 + $2 + $3 + $4;} END { printf "%.5f\n", sum } ')
        if [ $(echo "$PERCENT > 0.001" | bc) -eq 1 ]
        then
            echo $i
            #echo $i 1>&2
        fi
        echo -n . 1>&2
    done | tee "tmp.tmp"
    echo 1>&2
}

set +x
pdftk "${IN}" cat $(non_blank) output "$OUT"
