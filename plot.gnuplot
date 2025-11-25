set terminal pngcairo enhanced size 1400,1600 font 'CMU Serif,12'
set output ARG2

file = ARG1

set datafile separator ','
set key autotitle columnhead
set grid

set multiplot layout 3,2 title sprintf('Plot of {/"CMU Typewriter Text" %s}', file)
set key top left

set title 'Number of Clients over Time'
set xlabel 'time'
set ylabel '#clients'
plot file using 'time':'num_clients' with lines lw 2 title '#clients'

set title 'Task Latency over Time'
set xlabel 'time'
set ylabel 'latency'
plot file using 'time':'task_latency' with lines lw 2 title 'task latency'

unset multiplot
