package main

import (
	"log"
	"os"
	"photot/api"
	"photot/api/handler"
	"photot/helper/database"
)

func main() {
	hand := NewHandler()
	router := api.Router(hand)
	log.Printf("server is running...")
	log.Fatal(router.Run(":8080"))
}

func NewHandler() *handler.Handler {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("server is preparing for run...")

	imageDir := "./images"
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		err = os.MkdirAll(imageDir, 0755)
		if err != nil {
			log.Fatalf("Unable to create pictures folder: %v", err)
		}
		log.Printf("pictures folder is created: %s", imageDir)
	}

	db := database.NewImageDatabase()
	if err := db.LoadImages(imageDir); err != nil {
		log.Fatalf("Could not load images: %v", err)
	}
	return &handler.Handler{
		DB:       db,
		ImageDir: imageDir,
	}
}
