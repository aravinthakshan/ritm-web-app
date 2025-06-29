"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  MousePointer,
  Undo2,
  RotateCcw,
  Download,
  CheckCircle,
  Info,
  Loader2,
  ImageIcon,
  Zap,
} from "lucide-react"

interface CanvasState {
  image: HTMLImageElement | null
  scale: number
  offsetX: number
  offsetY: number
}

export default function InteractiveSegmentationDemo() {
  const [currentImage, setCurrentImage] = useState<HTMLImageElement | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState("Processing...")
  const [status, setStatus] = useState("No image loaded")
  const [clickCount, setClickCount] = useState(0)
  const [objectCount, setObjectCount] = useState(0)
  const [isDragOver, setIsDragOver] = useState(false)
  const [canvasState, setCanvasState] = useState<CanvasState>({
    image: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  })

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const drawImage = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !currentImage) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(
      currentImage,
      canvasState.offsetX,
      canvasState.offsetY,
      currentImage.width * canvasState.scale,
      currentImage.height * canvasState.scale,
    )
  }, [currentImage, canvasState])

  useEffect(() => {
    drawImage()
  }, [drawImage])

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setStatus("Please select an image file")
      return
    }

    setIsLoading(true)
    setLoadingMessage("Uploading image...")

    const formData = new FormData()
    formData.append("image", file)

    fetch("/api/upload_image", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          loadImage(data.image, data.width, data.height)
          setStatus("Image loaded successfully")
        } else {
          setStatus("Error: " + data.error)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error uploading image")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const loadImage = (imageData: string, width: number, height: number) => {
    const img = new Image()
    img.onload = () => {
      setCurrentImage(img)

      const canvas = canvasRef.current
      if (!canvas) return

      const canvasWidth = canvas.width
      const canvasHeight = canvas.height
      const scaleX = canvasWidth / width
      const scaleY = canvasHeight / height
      const scale = Math.min(scaleX, scaleY, 1)

      const offsetX = (canvasWidth - width * scale) / 2
      const offsetY = (canvasHeight - height * scale) / 2

      setCanvasState({
        image: img,
        scale,
        offsetX,
        offsetY,
      })
    }
    img.src = "data:image/png;base64," + imageData
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (!currentImage) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const imageX = Math.round((x - canvasState.offsetX) / canvasState.scale)
    const imageY = Math.round((y - canvasState.offsetY) / canvasState.scale)

    if (imageX >= 0 && imageX < currentImage.width && imageY >= 0 && imageY < currentImage.height) {
      addClick(imageX, imageY, e.button === 0)
    }
  }

  const handleCanvasContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (!currentImage) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const imageX = Math.round((x - canvasState.offsetX) / canvasState.scale)
    const imageY = Math.round((y - canvasState.offsetY) / canvasState.scale)

    if (imageX >= 0 && imageX < currentImage.width && imageY >= 0 && imageY < currentImage.height) {
      addClick(imageX, imageY, false)
    }
  }

  const addClick = (x: number, y: number, isPositive: boolean) => {
    setIsLoading(true)
    setLoadingMessage("Processing click...")

    fetch("/api/add_click", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        x: x,
        y: y,
        is_positive: isPositive,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          updateImage(data.image)
          setClickCount(data.clicks_count)
          setStatus("Click added successfully")
        } else {
          setStatus("Error: " + data.error)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error adding click")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const finishObject = () => {
    setIsLoading(true)
    setLoadingMessage("Finishing object...")

    fetch("/api/finish_object", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          updateImage(data.image)
          setObjectCount(data.object_count)
          setStatus("Object finished")
        } else {
          setStatus("Error: " + data.error)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error finishing object")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const undoClick = () => {
    setIsLoading(true)
    setLoadingMessage("Undoing click...")

    fetch("/api/undo_click", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          updateImage(data.image)
          setClickCount(data.clicks_count)
          setStatus("Click undone")
        } else {
          setStatus("Error: " + data.error)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error undoing click")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const resetClicks = () => {
    setIsLoading(true)
    setLoadingMessage("Resetting clicks...")

    fetch("/api/reset_clicks", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          updateImage(data.image)
          setClickCount(data.clicks_count)
          setStatus("Clicks reset")
        } else {
          setStatus("Error: " + data.error)
        }
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error resetting clicks")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const saveMask = () => {
    setIsLoading(true)
    setLoadingMessage("Saving mask...")

    fetch("/api/save_mask", {
      method: "POST",
    })
      .then((response) => {
        if (response.ok) {
          return response.blob()
        } else {
          throw new Error("Failed to save mask")
        }
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = "mask.png"
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
        setStatus("Mask saved successfully")
      })
      .catch((error) => {
        console.error("Error:", error)
        setStatus("Error saving mask")
      })
      .finally(() => {
        setIsLoading(false)
      })
  }

  const updateImage = (imageData: string) => {
    const img = new Image()
    img.onload = () => {
      setCurrentImage(img)
      setCanvasState((prev) => ({ ...prev, image: img }))
    }
    img.src = "data:image/png;base64," + imageData
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  return (
    <div className="min-h-screen bg-neutral-900 text-white">
      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center">
          <Card className="w-80 bg-neutral-800 border-neutral-700">
            <CardContent className="pt-6">
              <div className="flex flex-col items-center space-y-4">
                <Loader2 className="h-8 w-8 animate-spin text-neutral-300" />
                <p className="text-sm text-neutral-400">{loadingMessage}</p>
                <Progress value={undefined} className="w-full" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="container mx-auto p-4 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            {/* <div className="p-3 bg-gradient-to-br from-neutral-700 to-neutral-900 rounded-xl border border-neutral-600">
              <Zap className="h-8 w-8 text-neutral-300" />
            </div> */}
            <h1 className="text-4xl font-bold bg-gradient-to-r from-neutral-200 to-neutral-400 bg-clip-text text-transparent">
              Interactive Segmentation Demo
            </h1>
          </div>
          <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
            Click on the image to add positive (left click) or negative (right click) points for intelligent
            segmentation
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Left Panel - Upload and Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Upload Section */}
            <Card className="bg-neutral-800 border-neutral-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-neutral-200">
                  <Upload className="h-5 w-5" />
                  Upload Image
                </CardTitle>
                <CardDescription className="text-neutral-400">
                  Drag and drop an image or click to browse
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer hover:border-neutral-500 hover:bg-neutral-700/50 ${
                    isDragOver ? "border-neutral-500 bg-neutral-700/50" : "border-neutral-600"
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <ImageIcon className="h-12 w-12 mx-auto mb-4 text-neutral-500" />
                  <p className="text-sm text-neutral-400 mb-4">Drop your image here or click to browse</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-neutral-600 text-neutral-300 hover:bg-neutral-700 bg-transparent"
                  >
                    Choose Image
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) handleFileSelect(file)
                    }}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Segmentation Controls - MOVED HERE */}
            <Card className="bg-neutral-800 border-neutral-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-neutral-200">
                  <MousePointer className="h-5 w-5" />
                  Segmentation Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  onClick={finishObject}
                  disabled={!currentImage}
                  className="w-full bg-neutral-700 hover:bg-neutral-600 text-neutral-200"
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Finish Object
                </Button>
                <div className="grid grid-cols-2 gap-3">
                  <Button
                    onClick={undoClick}
                    disabled={!currentImage}
                    variant="outline"
                    className="border-neutral-600 text-neutral-300 hover:bg-neutral-700 bg-transparent"
                  >
                    <Undo2 className="h-4 w-4 mr-2" />
                    Undo
                  </Button>
                  <Button
                    onClick={resetClicks}
                    disabled={!currentImage}
                    variant="outline"
                    className="border-neutral-600 text-neutral-300 hover:bg-neutral-700 bg-transparent"
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Status */}
            <Card className="bg-neutral-800 border-neutral-700">
              <CardHeader>
                <CardTitle className="text-sm text-neutral-200">Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm">
                  <p className="font-medium text-neutral-300">{status}</p>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-400">Clicks:</span>
                  <Badge variant="secondary" className="bg-neutral-700 text-neutral-300">
                    {clickCount}
                  </Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-neutral-400">Objects:</span>
                  <Badge variant="secondary" className="bg-neutral-700 text-neutral-300">
                    {objectCount}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Center Panel - Canvas */}
          <div className="lg:col-span-3">
            <Card className="h-full bg-neutral-800 border-neutral-700">
              <CardHeader>
                <CardTitle className="text-neutral-200">Canvas</CardTitle>
                <CardDescription className="text-neutral-400">Click on the image to start segmentation</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="border rounded-lg overflow-hidden bg-neutral-900 border-neutral-600 flex items-center justify-center min-h-[500px]">
                  <canvas
                    ref={canvasRef}
                    width={800}
                    height={600}
                    className="max-w-full max-h-full cursor-crosshair"
                    onClick={handleCanvasClick}
                    onContextMenu={handleCanvasContextMenu}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom Panel - Controls */}
        <div className="grid md:grid-cols-2 gap-6 mt-6">
          {/* Export */}
          <Card className="bg-neutral-800 border-neutral-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-neutral-200">
                <Download className="h-5 w-5" />
                Export
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Button
                onClick={saveMask}
                disabled={!currentImage}
                className="w-full bg-neutral-700 hover:bg-neutral-600 text-neutral-200"
              >
                <Download className="h-4 w-4 mr-2" />
                Save Mask
              </Button>
            </CardContent>
          </Card>

          {/* Instructions */}
          <Card className="bg-neutral-800 border-neutral-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-neutral-200">
                <Info className="h-4 w-4" />
                Instructions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-neutral-400">
                <p>
                  <strong className="text-neutral-300">Left Click:</strong> Add positive point (foreground)
                </p>
                <p>
                  <strong className="text-neutral-300">Right Click:</strong> Add negative point (background)
                </p>
                <p>
                  <strong className="text-neutral-300">Finish Object:</strong> Complete current segmentation
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
