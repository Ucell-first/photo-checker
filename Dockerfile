FROM golang:1.24.1 AS builder

WORKDIR /app

COPY . .

RUN go mod download

COPY .env .

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o myapp .

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/myapp .
COPY --from=builder /app/.env .

EXPOSE 8080

CMD ["./myapp"]