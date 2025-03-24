# Photo Checker

A Go-based web service for managing and checking photos.

## Overview

Photo Checker is a web service that provides an API for managing images. It includes functionality for storing images and maintaining an image database. It will check photos using ml and hashing.

## Features

- RESTful API endpoints for image management
- Automatic image directory creation and management
- Built-in image database
- Server runs on port 8080

## Prerequisites

- Go 1.x
- Gin web framework

## API
1. Recognize Image
- Endpoint: /recognize
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - image (file, required): Image to recognize
  - threshold (number, optional): Similarity threshold (0-100)
- Response:
{
  "processing_time_ms": 123,
  "similarity": 85.5,
  "method": "ml/hash",
  "result": "OK/NOT OK",
  "matched_image": "filename.ext"
}


2. Toggle ML Recognition
- Endpoint: /admin/toggle-ml
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
- enable (string): "true" or "false"
- Response:
{
  "message": "ML enabled/disabled",
  "status": "enabled/disabled"
}

3. Health Check
- Endpoint: /admin/hello
- Method: GET
- Response:
{
  "message": "Hello, world"
}

4. Add image to image file
- **URL:** `http://localhost:8080/admin/add`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Form Parameter:** `file` (image file)
- **Description:** Uploads an image file to the server's `images` directory
- **Response:** 
  - Success: `200 OK` with message
  - Error: `400 Bad Request` if file is invalid