set terminal pngcairo size 1400,1600 enhanced font 'Iosevka,12'
set output ARG2

set datafile separator ','
set key autotitle columnhead
set grid

set multiplot layout 3,2 title "Performance Metrics vs Number of Clients"

file = ARG1

set title 'Mean Queue Size'
set xlabel 'num\_clients'
set ylabel 'queue\_size\_mean'
plot file using 'num_clients':'queue_size_mean' with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_mean'

set title 'Min/Max Queue Size'
set xlabel 'num\_clients'
set ylabel 'queue\_size\_max'
plot file using 'num_clients':'queue_size_max' with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_max', \
     file using 'num_clients':'queue_size_min' with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_min'

# Plot 5: latency_mean
set title 'Mean Latency'
set xlabel 'num\_clients'
set ylabel 'latency\_mean'
plot file using 'num_clients':'latency_mean' with linespoints pt 7 ps 1 lw 2 title 'latency\_mean'

set title 'Min/Max Latency'
set xlabel 'num\_clients'
set ylabel 'latency\_max'
plot file using 'num_clients':'latency_max' with linespoints pt 7 ps 1 lw 2 title 'latency\_max', \
     file using 'num_clients':'latency_min' with linespoints pt 7 ps 1 lw 2 title 'latency\_min'

set title 'Failure Rate'
set xlabel 'num\_clients'
set ylabel 'fail\_rate'
plot file using 'num_clients':'fail_rate' with linespoints pt 7 ps 1 lw 2 title 'fail\_rate'

set title 'Total and Failed Tasks'
set xlabel 'num\_clients'
set ylabel 'tasks'
plot file using 'num_clients':'tasks_total' with linespoints pt 7 ps 1 lw 2 title 'tasks\_total', \
     file using 'num_clients':'tasks_failed' with linespoints pt 5 ps 1 lw 2 title 'tasks\_failed'

unset multiplot
