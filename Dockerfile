FROM docker.io/library/rust:1-slim AS builder

WORKDIR /build
COPY . .
RUN cargo build --release --locked \
    && strip target/release/anthropic-proxy

FROM docker.io/library/debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/anthropic-proxy /usr/local/bin/

EXPOSE 3000

ENTRYPOINT ["anthropic-proxy"]
