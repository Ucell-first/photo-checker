include .env
export $(shell sed 's/=.*//' .env)

CURRENT_DIR=$(shell pwd)
PDB_URL := postgres://$(DB_USER):$(DB_PASSWORD)@localhost:$(DB_PORT)/$(DB_NAME)?sslmode=disable

proto-gen:
	./scripts/gen-proto.sh ${CURRENT_DIR}

mig-up:
	migrate -path migrations -database '${PDB_URL}' -verbose up

mig-down:
	migrate -path migrations -database '${PDB_URL}' -verbose down

mig-force:
	migrate -path migrations -database '${PDB_URL}' -verbose force 1

create_mig:
	@echo "Enter file name: "; \
	read filename; \
	migrate create -ext sql -dir migrations -seq $$filename

swag:
	~/go/bin/swag init -g ./api/router.go -o ./api/docs


	
run:
	go run cmd/main.go
git:
	@echo "Enter commit name: "; \
	read commitname; \
	git add .; \
	git commit -m "$$commitname"; \
	if ! git push origin main; then \
		echo "Push failed. Attempting to merge and retry..."; \
		$(MAKE) git-merge; \
		git add .; \
		git commit -m "$$commitname"; \
		git push origin main; \
	fi

git-merge:
	git fetch origin; \
	git merge origin/main

google:
	mkdir -p protos/google/api
	mkdir -p protos/protoc-gen-openapiv2/options
	curl -o protos/google/api/annotations.proto https://raw.githubusercontent.com/googleapis/googleapis/master/google/api/annotations.proto
	curl -o protos/google/api/http.proto https://raw.githubusercontent.com/googleapis/googleapis/master/google/api/http.proto
	curl -o protos/protoc-gen-openapiv2/options/annotations.proto https://raw.githubusercontent.com/grpc-ecosystem/grpc-gateway/main/protoc-gen-openapiv2/options/annotations.proto
	curl -o protos/protoc-gen-openapiv2/options/openapiv2.proto https://raw.githubusercontent.com/grpc-ecosystem/grpc-gateway/main/protoc-gen-openapiv2/options/openapiv2.proto


proto:
	rm -f generated/**/*.go
	rm -f doc/swagger/*.swagger.json
	mkdir -p generated
	mkdir -p doc/swagger
	echo '<!DOCTYPE html><html><head><title>API Documentation</title><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" type="text/css" href="//unpkg.com/swagger-ui-dist@3/swagger-ui.css" /></head><body><div id="swagger-ui"></div><script src="//unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script><script>const ui = SwaggerUIBundle({url: "swagger_docs.swagger.json",dom_id: "#swagger-ui",deepLinking: true,presets: [SwaggerUIBundle.presets.apis,SwaggerUIBundle.SwaggerUIStandalonePreset],plugins: [SwaggerUIBundle.plugins.DownloadUrl],})</script></body></html>' > doc/swagger/index.html
	protoc \
		--proto_path=protos --go_out=generated --go_opt=paths=source_relative \
		--go-grpc_out=generated --go-grpc_opt=paths=source_relative \
		--grpc-gateway_out=generated --grpc-gateway_opt=paths=source_relative \
		--openapiv2_out=doc/swagger --openapiv2_opt=allow_merge=true,merge_file_name=swagger_docs,use_allof_for_refs=true,disable_service_tags=true,json_names_for_fields=false \
		--validate_out="lang=go,paths=source_relative:generated" \
			protos/**/*.proto