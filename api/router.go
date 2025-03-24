package api

import (
	_ "photot/api/docs"
	"photot/api/handler"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title Photo Recognition API
// @version 1.1
// @description API for image recognition using ML and perceptual hashing
// @BasePath /
func Router(hand *handler.Handler) *gin.Engine {
	r := gin.New()
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	r.POST("/recognize", hand.RecognizeHandler)

	admin := r.Group("/admin")
	{
		admin.POST("/add", hand.AddImageHandler)
		admin.GET("/hello", hand.Hello)
		admin.POST("/toggle-ml", hand.ToggleMLHandler)
	}
	return r
}
