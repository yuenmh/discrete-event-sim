CSVS=$(wildcard results/*.csv)
PLOTS=$(patsubst results/%.csv,plots/%.png,$(CSVS))

plots: $(PLOTS)

plots/%.png: results/%.csv
	gnuplot -c plot.gnuplot $< $@
