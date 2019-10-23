release:
	cargo build --release

debug:
	cargo build

clean:
	cargo clean

test:
	cargo test -- --nocapture
