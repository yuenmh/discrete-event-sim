CSVS=$(wildcard results/*.csv)
PLOTS=$(patsubst results/%.csv,plots/%.png,$(CSVS))
SCRIPT=plot.gnuplot

plots: $(SCRIPT) $(PLOTS)

plots/%.png: results/%.csv
	gnuplot -c $(SCRIPT) $< $@
