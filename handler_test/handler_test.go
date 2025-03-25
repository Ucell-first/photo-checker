package handler_test

import (
	"bytes"
	"image"
	"image/color"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"photot/api/handler"
	"photot/helper/database"

	"github.com/disintegration/imaging"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestHandler(t *testing.T) {
	// Test uchun vaqtni sozlash
	now := time.Now().UnixNano()
	testDir := "./test_images_" + strconv.FormatInt(now, 10) // 10 - decimal base
	_ = os.Mkdir(testDir, 0755)
	defer os.RemoveAll(testDir)

	// Har bir test uchun yangi handler yaratish
	newHandler := func() *handler.Handler {
		db := database.NewImageDatabase()
		return &handler.Handler{
			DB:       db,
			ImageDir: testDir,
		}
	}

	t.Run("TestAddImageHandler", func(t *testing.T) {
		h := newHandler()

		// Unique filename
		filename := "test_" + strconv.FormatInt(now, 10) + ".png"
		img := createTestImage()

		// So'rovni tayyorlash
		body := &bytes.Buffer{}
		writer := multipart.NewWriter(body)
		part, _ := writer.CreateFormFile("image", filename)
		imaging.Encode(part, img, imaging.PNG)
		writer.Close()

		req, _ := http.NewRequest("POST", "/admin/add", body)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		resp := httptest.NewRecorder()

		// Kontekstni yaratish
		ctx, _ := gin.CreateTestContext(resp)
		ctx.Request = req
		h.AddImageHandler(ctx)

		// Tekshirishlar
		assert.Equal(t, http.StatusOK, resp.Code)
		assert.Contains(t, resp.Body.String(), "image added successfully")
	})

	t.Run("TestDuplicateImage", func(t *testing.T) {
		h := newHandler()
		filename := "duplicate_test.png"

		// Birinchi marta qo'shish
		img := createTestImage()
		addImage(h, filename, t)

		// Ikkinchi marta qo'shish
		body := &bytes.Buffer{}
		writer := multipart.NewWriter(body)
		part, _ := writer.CreateFormFile("image", filename)
		imaging.Encode(part, img, imaging.PNG)
		writer.Close()

		req, _ := http.NewRequest("POST", "/admin/add", body)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		resp := httptest.NewRecorder()

		ctx, _ := gin.CreateTestContext(resp)
		ctx.Request = req
		h.AddImageHandler(ctx)

		assert.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "already exists")
	})

	t.Run("TestToggleMLHandler", func(t *testing.T) {
		h := newHandler()

		// ML ni o'chirish
		body := &bytes.Buffer{}
		writer := multipart.NewWriter(body)
		writer.WriteField("enable", "false")
		writer.Close()

		req, _ := http.NewRequest("POST", "/admin/toggle-ml", body)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		resp := httptest.NewRecorder()

		ctx, _ := gin.CreateTestContext(resp)
		ctx.Request = req
		h.ToggleMLHandler(ctx)

		assert.Equal(t, http.StatusOK, resp.Code)
		assert.Contains(t, resp.Body.String(), "ML disabled")
	})
}

// Yordamchi funksiyalar
func createTestImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{
				R: uint8((x + y) % 256),
				G: uint8((x * y) % 256),
				B: uint8((x - y) % 256),
				A: 255,
			})
		}
	}
	return img
}

func addImage(h *handler.Handler, filename string, t *testing.T) {
	img := createTestImage()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("image", filename)
	imaging.Encode(part, img, imaging.PNG)
	writer.Close()

	req, _ := http.NewRequest("POST", "/admin/add", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp := httptest.NewRecorder()

	ctx, _ := gin.CreateTestContext(resp)
	ctx.Request = req
	h.AddImageHandler(ctx)

	assert.Equal(t, http.StatusOK, resp.Code, "Initial image add failed")
}
