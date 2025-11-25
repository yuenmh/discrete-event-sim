set terminal pngcairo enhanced size 1400,1600 font 'CMU Serif,12'
set output ARG2

file = ARG1

set datafile separator ','
set key autotitle columnhead
set grid

set multiplot layout 3,2 title sprintf('Plot of {/"CMU Typewriter Text" %s}', file)
set key top left

set yrange [0:*]

set title 'Number of Clients'
set xlabel 'time'
set ylabel '#clients'
plot file using 'time':'num_clients' with lines lw 2 title '#clients'

set title 'Task Latency'
set xlabel 'start time'
set ylabel 'latency'
plot file using 'time':'smoothed_task_latency' with lines lw 2 title 'task latency'

set title 'Queue Size'
set xlabel 'time'
set ylabel 'queue size'
plot file using 'time':'smoothed_queue_size' with lines lw 2 title 'queue size'

set title 'Total Completed Tasks'
set xlabel 'time'
set ylabel '#completed tasks'
plot file using 'time':'total_successes' with lines lw 2 title 'successes', \
     file using 'time':'total_failures' with lines lw 2 title 'failures'

unset multiplot
