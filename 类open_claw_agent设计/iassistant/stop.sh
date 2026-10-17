#!/bin/bash

ROOT=$(cd `dirname $0`; pwd)

SVC_NAME='iassistant'

PPIDs=()
if [[ ${MZ_ENV:-"dev"} == 'dev' ]]; then
    PPIDs=`pgrep -u $USER -fa "${ROOT}" | grep main | grep MZ_ENV | cut -d' ' -f1`
else
    PPIDs=`pgrep -fa "${ROOT}" | grep main | grep MZ_ENV | cut -d' ' -f1`
fi

for ppid in ${PPIDs[@]}; do
    for pid in `pgrep -P $ppid -a | cut -d' ' -f1`; do
        kill -INT $pid
    done
    echo "Shutdown fastapi process (PID $ppid) successfully"
done

PPIDs=()
if [[ ${MZ_ENV:-"dev"} == 'dev' ]]; then
    PPIDs=`pgrep -u $USER -fa "${ROOT}" | grep worker | grep MZ_ENV | cut -d' ' -f1`
else
    PPIDs=`pgrep -fa "${ROOT}" | grep worker | grep MZ_ENV | cut -d' ' -f1`
fi

for ppid in ${PPIDs[@]}; do
    for pid in `pgrep -P $ppid -a | cut -d' ' -f1`; do
        kill -INT $pid
    done
    echo "Shutdown dramatiq process (PID $ppid) successfully"
done

