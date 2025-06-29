# Interactive Segmentation Demo - Integrated Setup

This is an integrated setup combining a **Flask backend** with a **Next.js frontend** for interactive image segmentation.

## 🏗️ Architecture

```
┌─────────────────┐    API Calls    ┌─────────────────┐
│   Next.js       │ ──────────────► │   Flask         │
│   Frontend      │                 │   Backend       │
│   (Port 3000)   │ ◄────────────── │   (Port 5000)   │
└─────────────────┘                 └─────────────────┘
```

- **Frontend**: Next.js with TypeScript, Tailwind CSS, and shadcn/ui components
- **Backend**: Flask with your trained segmentation model
- **Communication**: REST API with CORS support

## 🚀 Quick Start

### Option 1: Using the Integrated Script (Recommended)

```bash
# From the project root directory
cd web_demo
python start_integrated_demo.py
```

This will automatically:
- Install dependencies
- Start the Flask backend on port 5000
- Start the Next.js frontend on port 3000
- Open your browser to `http://localhost:3000`

### Option 2: Manual Setup

#### 1. Install Dependencies

```bash
cd web_demo

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd interactive-segmentation-demo
npm install
cd ..
```

#### 2. Start the Backend

```bash
# In one terminal
python app.py
```

The Flask backend will start on `http://localhost:5000`

#### 3. Start the Frontend

```bash
# In another terminal
cd interactive-segmentation-demo
npm run dev
```

The Next.js frontend will start on `http://localhost:3000`

## 🎯 Features

### Frontend Features
- **Modern UI**: Built with Next.js, TypeScript, and Tailwind CSS
- **Drag & Drop**: Upload images by dragging them onto the interface
- **Interactive Canvas**: Click to add positive/negative points
- **Real-time Updates**: See segmentation results immediately
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Loading States**: Visual feedback during processing
- **Error Handling**: User-friendly error messages

### Backend Features
- **Real Model**: Uses your trained `hrnet18_cocolvis_itermask_3p` model
- **API Endpoints**: RESTful API for all segmentation operations
- **CORS Support**: Cross-origin requests enabled
- **Image Processing**: Automatic format conversion and validation
- **State Management**: Maintains session state for undo/redo

## 🔧 API Endpoints

All endpoints are available at `/api/` prefix:

- `POST /api/upload_image` - Upload and process an image
- `POST /api/add_click` - Add a click point (positive or negative)
- `POST /api/finish_object` - Complete current segmentation
- `POST /api/undo_click` - Remove last click
- `POST /api/reset_clicks` - Reset all clicks for current object
- `POST /api/save_mask` - Download segmentation mask
- `GET /api/model_info` - Get information about the loaded model

## 📁 File Structure

```
web_demo/
├── app.py                              # Flask backend
├── model_loader.py                     # Model loading utilities
├── requirements.txt                    # Python dependencies
├── start_integrated_demo.py           # Integrated startup script
├── package.json                       # Node.js package config
├── README_INTEGRATED.md               # This file
└── interactive-segmentation-demo/     # Next.js frontend
    ├── app/
    │   └── page.tsx                   # Main application page
    ├── components/                    # React components
    ├── package.json                   # Frontend dependencies
    └── next.config.mjs               # Next.js configuration
```

## 🛠️ Development

### Backend Development

```bash
# Start Flask in debug mode
cd web_demo
python app.py
```

### Frontend Development

```bash
# Start Next.js development server
cd web_demo/interactive-segmentation-demo
npm run dev
```

### API Proxy Configuration

The Next.js frontend is configured to proxy API requests to the Flask backend:

```javascript
// next.config.mjs
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:5000/:path*',
    },
  ]
}
```

This means frontend API calls like `/api/upload_image` are automatically forwarded to `http://localhost:5000/upload_image`.

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing processes
   pkill -f "python app.py"
   pkill -f "next dev"
   ```

2. **CORS Errors**
   - Ensure `flask-cors` is installed: `pip install flask-cors`
   - Check that CORS is enabled in `app.py`

3. **Model Loading Issues**
   - Verify the model path in `model_loader.py`
   - Check that all dependencies are installed

4. **Frontend Build Issues**
   ```bash
   cd interactive-segmentation-demo
   rm -rf node_modules package-lock.json
   npm install
   ```

### Debug Mode

To run in debug mode:

```bash
# Backend debug
cd web_demo
FLASK_ENV=development python app.py

# Frontend debug
cd interactive-segmentation-demo
npm run dev
```

## 🚀 Deployment

### Production Build

```bash
# Build the frontend
cd web_demo/interactive-segmentation-demo
npm run build

# Start production server
npm start
```

### Docker Deployment

Create a `Dockerfile` for containerized deployment:

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📝 Configuration

### Model Configuration

Edit `model_loader.py` to change model settings:

```python
def get_model_config():
    return {
        'model_type': 'pytorch',
        'model_path': '/path/to/your/model.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
```

### Frontend Configuration

Edit `interactive-segmentation-demo/next.config.mjs` to change proxy settings:

```javascript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://your-backend-url:5000/:path*',
    },
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the integration
5. Submit a pull request

## 📄 License

This project is based on the original interactive segmentation demo. Please refer to the original project's license for usage terms. 