run:
	cargo build --release && ./target/release/mnistnet

w:
	pandoc -f markdown-implicit_figures -o out.pdf writeup.md && open out.pdf


bt:
	cargo build --release && RUST_BACKTRACE=1 ./target/release/mnistnet
