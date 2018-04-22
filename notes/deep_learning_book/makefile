SOURCES = $(wildcard sources/*.md)
OUTS = $(patsubst sources/%.md, out/%.pdf, $(SOURCES))

all: $(SOURCES)
	@if [ ! -d "out"  ]; then mkdir "out"; fi
	@make $(OUTS)

$(OUTS): out/%.pdf : sources/%.md
	pandoc $< -o $@

.PHONY: clean
clean:
	rm -f out/*
