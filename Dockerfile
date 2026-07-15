FROM rust:1-bookworm AS builder

WORKDIR /build
COPY . .
RUN cargo build --release --locked \
    && strip target/release/anthropic-proxy

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/anthropic-proxy /usr/local/bin/

EXPOSE 3000

ENTRYPOINT ["anthropic-proxy"]
