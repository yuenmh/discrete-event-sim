set terminal pngcairo size 1400,1600 enhanced font 'Iosevka,12'
set output 'plot.png'

set datafile separator ','
set key autotitle columnhead
set grid

set multiplot layout 3,2 title "Performance Metrics vs Number of Clients"

set title 'Mean Queue Size'
set xlabel 'num\_clients'
set ylabel 'queue\_size\_mean'
plot 'result.csv' using 1:3 with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_mean'

set title 'Min/Max Queue Size'
set xlabel 'num\_clients'
set ylabel 'queue\_size\_max'
plot 'result.csv' using 1:4 with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_max', \
     'result.csv' using 1:5 with linespoints pt 7 ps 1 lw 2 title 'queue\_size\_min'

# Plot 5: latency_mean
set title 'Mean Latency'
set xlabel 'num\_clients'
set ylabel 'latency\_mean'
plot 'result.csv' using 1:6 with linespoints pt 7 ps 1 lw 2 title 'latency\_mean'

set title 'Min/Max Latency'
set xlabel 'num\_clients'
set ylabel 'latency\_max'
plot 'result.csv' using 1:7 with linespoints pt 7 ps 1 lw 2 title 'latency\_max', \
     'result.csv' using 1:8 with linespoints pt 7 ps 1 lw 2 title 'latency\_min'

set title 'Failure Rate'
set xlabel 'num\_clients'
set ylabel 'fail\_rate'
plot 'result.csv' using 1:11 with linespoints pt 7 ps 1 lw 2 title 'fail\_rate'

set title 'Total and Failed Tasks'
set xlabel 'num\_clients'
set ylabel 'tasks'
plot 'result.csv' using 1:9 with linespoints pt 7 ps 1 lw 2 title 'tasks\_total', \
     'result.csv' using 1:10 with linespoints pt 5 ps 1 lw 2 title 'tasks\_failed'

unset multiplot
