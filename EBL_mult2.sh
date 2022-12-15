#!/bin/sh
. /data/magic/users-ifae/rgrau/pythonenv/bin/activate
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $1
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+1000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+2000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+3000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+4000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+5000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+6000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+7000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+8000))
python3 /data/magic/users-ifae/rgrau/EBL-splines/EBL_mult2_1.py $(($1+9000))
