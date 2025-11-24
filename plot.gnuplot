set terminal pngcairo enhanced size 1400,1600 font 'CMU Serif,12'
set output ARG2

file = ARG1

set datafile separator ','
set key autotitle columnhead
set grid

set multiplot layout 3,2 title sprintf('Plot of {/"CMU Typewriter Text" %s}', file)
set key top left

set title 'Mean Queue Size'
set xlabel 'num clients'
set ylabel 'queue size'
plot file using 'num_clients':'queue_size_mean' with linespoints pt 7 ps 1 lw 2 title 'mean queue size'

set title 'Min/Max Queue Size'
set xlabel 'num clients'
set ylabel 'queue size'
plot file using 'num_clients':'queue_size_max' with linespoints pt 7 ps 1 lw 2 title 'max', \
     file using 'num_clients':'queue_size_min' with linespoints pt 5 ps 1 lw 2 title 'min'

# Plot 5: latency_mean
set title 'Mean Latency'
set xlabel 'num clients'
set ylabel 'latency'
plot file using 'num_clients':'latency_mean' with linespoints pt 7 ps 1 lw 2 title 'mean latency'

set title 'Min/Max Latency'
set xlabel 'num clients'
set ylabel 'latency'
plot file using 'num_clients':'latency_max' with linespoints pt 7 ps 1 lw 2 title 'max', \
     file using 'num_clients':'latency_min' with linespoints pt 5 ps 1 lw 2 title 'min'

set title 'Failure Rate'
set xlabel 'num clients'
set ylabel 'failure rate'
plot file using 'num_clients':'fail_rate' with linespoints pt 7 ps 1 lw 2 title 'failure rate'

set title 'Total and Failed Tasks'
set xlabel 'num clients'
set ylabel 'tasks'
plot file using 'num_clients':'tasks_total' with linespoints pt 7 ps 1 lw 2 title 'total', \
     file using 'num_clients':'tasks_failed' with linespoints pt 5 ps 1 lw 2 title 'failed'

unset multiplot
