#!/bin/bash
function lock(){
echo `date +'%F %T.%3N'`"----------------start exeute "$1"-------------"
./$1 wrmem 0x142600070 0x0
./$1 turbocfg 2 1 0
./$1 turbocfg 2 2 0
./$1 turbocfg 2 3 0
./$1 turbocfg 2 4 0
./$1 turbocfg 2 5 0
./$1 dfsaic 1800
}
for i in {0..7}
do
       if [ $i -eq 0 ]
       then
	       lock lptest
       else
	       cp lptest lptest$i
	       lock lptest$i
       fi
       sleep 5

done

