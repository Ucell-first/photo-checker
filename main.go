// @title Image Recognition API
// @version 1.0
// @description Rasmlarni solishtirish uchun API servis
// @termsOfService http://swagger.io/terms/
// @contact.name API Support
// @contact.email support@imageapi.com
// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html
// @host localhost:8080
// @BasePath /
// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name Authorization
package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/disintegration/imaging"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	_ "photot/docs"
)

var existingHashes []string

// computeHash rasm hashini hisoblaydi
func computeHash(img image.Image) string {
	resized := imaging.Resize(img, 8, 8, imaging.Lanczos)
	gray := imaging.Grayscale(resized)

	var sum uint64
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			c := gray.At(x, y).(color.NRGBA) // color.Gray o‘rniga color.NRGBA
			sum += uint64(c.R)               // Gray qiymat sifatida c.R ishlatamiz
		}
	}
	avg := uint8(sum / 64)

	var hash strings.Builder
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			c := gray.At(x, y).(color.NRGBA) // color.Gray o‘rniga color.NRGBA
			if c.R >= avg {
				hash.WriteString("1")
			} else {
				hash.WriteString("0")
			}
		}
	}
	return hash.String()
}

// hammingDistance hash'lar orasidagi masofani hisoblaydi
func hammingDistance(hash1, hash2 string) int {
	if len(hash1) != len(hash2) {
		return -1
	}
	distance := 0
	for i := 0; i < len(hash1); i++ {
		if hash1[i] != hash2[i] {
			distance++
		}
	}
	return distance
}

// @Summary Recognize image
// @Description Foydalanuvchi yuborgan rasmni bazadagi rasmlar bilan solishtirish
// @Tags Image Recognition
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Tekshiriladigan rasm"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /recognize [post]
func recognizeHandler(c *gin.Context) {
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Rasm fayli topilmadi"})
		return
	}
	defer file.Close()

	if header.Size > 10<<20 { // 10MB cheklash
		c.JSON(http.StatusBadRequest, gin.H{"error": "Fayl hajmi 10MB dan katta"})
		return
	}

	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Faylni o'qib bo'lmadi"})
		return
	}

	img, err := imaging.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Noto'g'ri rasm formati"})
		return
	}

	uploadedHash := computeHash(img)
	threshold := 5
	ok := false

	for _, existingHash := range existingHashes {
		if hammingDistance(uploadedHash, existingHash) <= threshold {
			ok = true
			break
		}
	}

	if ok {
		c.JSON(http.StatusOK, gin.H{"result": "OK"})
	} else {
		c.JSON(http.StatusOK, gin.H{"result": "NOT OK"})
	}
}

func loadExistingHashes() {
	imageDir := "./images"
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		log.Fatal("Images papkasi mavjud emas")
	}

	files, err := os.ReadDir(imageDir)
	if err != nil {
		log.Fatal("Papkani o'qib bo'lmadi: ", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}
		path := filepath.Join(imageDir, file.Name())
		img, err := imaging.Open(path)
		if err != nil {
			log.Printf("%s faylni ochib bo'lmadi: %v", path, err)
			continue
		}
		existingHashes = append(existingHashes, computeHash(img))
	}
}

func main() {
	loadExistingHashes()

	r := gin.Default()

	// Swagger UI
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// API endpoints
	r.POST("/recognize", recognizeHandler)

	fmt.Println("Server 8080 portida ishga tushdi...")
	if err := r.Run(":8080"); err != nil {
		log.Fatal("Serverni ishga tushirib bo'lmadi: ", err)
	}
}
