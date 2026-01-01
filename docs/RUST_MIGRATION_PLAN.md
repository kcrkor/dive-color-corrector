# Rust + Tauri Migration Plan

## Overview

This document outlines the plan to migrate Dive Color Corrector from Python to Rust with a Tauri-based desktop interface. The migration aims to achieve:

- Native performance for image/video processing
- Smaller binary size and faster startup
- Cross-platform desktop app without Python runtime dependency
- Memory safety guarantees from Rust

## Target Architecture

```
dive-color-corrector-rs/
├── src-tauri/                    # Rust backend
│   ├── src/
│   │   ├── main.rs               # Tauri entry point
│   │   ├── lib.rs                # Library root
│   │   ├── commands/             # Tauri command handlers
│   │   │   ├── mod.rs
│   │   │   ├── image.rs          # Image processing commands
│   │   │   └── video.rs          # Video processing commands
│   │   ├── core/                 # Core processing logic
│   │   │   ├── mod.rs
│   │   │   ├── color/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── constants.rs
│   │   │   │   ├── filter.rs
│   │   │   │   └── hue.rs
│   │   │   ├── correction.rs
│   │   │   └── processing/
│   │   │       ├── mod.rs
│   │   │       ├── image.rs
│   │   │       └── video.rs
│   │   └── models/               # ML model integration
│   │       ├── mod.rs
│   │       └── sesr.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
├── src/                          # Frontend (web)
│   ├── index.html
│   ├── main.ts
│   ├── App.vue                   # or React/Svelte
│   ├── components/
│   │   ├── FileSelector.vue
│   │   ├── Preview.vue
│   │   └── ProgressBar.vue
│   └── styles/
├── package.json
└── vite.config.ts
```

## Migration Phases

### Phase 1: Project Setup and Core Types

**Deliverables:**
- Initialize Tauri project with Vite + Vue/React frontend
- Define core Rust types and constants
- Set up CI/CD for multi-platform builds

**Tasks:**

1. Create new Tauri project
   ```bash
   npm create tauri-app@latest dive-color-corrector-rs
   cd dive-color-corrector-rs
   cargo add image ndarray rayon
   ```

2. Define core constants in `src-tauri/src/core/color/constants.rs`:
   ```rust
   pub const THRESHOLD_RATIO: f32 = 2000.0;
   pub const MIN_AVG_RED: f32 = 60.0;
   pub const MAX_HUE_SHIFT: i32 = 120;
   pub const BLUE_MAGIC_VALUE: f32 = 1.2;
   pub const SAMPLE_SECONDS: u32 = 2;
   ```

3. Define image matrix types using `ndarray`:
   ```rust
   use ndarray::{Array2, Array3};

   pub type RgbImage = Array3<u8>;      // Shape: (H, W, 3)
   pub type FilterMatrix = [f32; 20];
   ```

### Phase 2: Core Algorithm Implementation

**Deliverables:**
- Port hue shifting algorithm
- Port filter matrix computation
- Port filter application with SIMD optimization

**Key Crates:**
- `image` - Image I/O
- `ndarray` - N-dimensional arrays
- `rayon` - Parallel iteration
- `wide` or `packed_simd` - SIMD operations (optional)

**Tasks:**

1. Implement `hue_shift_red` in `src-tauri/src/core/color/hue.rs`:
   ```rust
   pub fn hue_shift_red(mat: &Array3<f32>, h: f32) -> Array3<f32> {
       let u = (h * std::f32::consts::PI / 180.0).cos();
       let w = (h * std::f32::consts::PI / 180.0).sin();

       // Vectorized computation using ndarray
       let r = &mat.slice(s![.., .., 0]);
       let g = &mat.slice(s![.., .., 1]);
       let b = &mat.slice(s![.., .., 2]);

       // ... transformation logic
   }
   ```

2. Implement `apply_filter` with parallel processing:
   ```rust
   use rayon::prelude::*;

   pub fn apply_filter(mat: &Array3<u8>, filter: &FilterMatrix) -> Array3<u8> {
       let (h, w, _) = mat.dim();
       let mut result = Array3::<u8>::zeros((h, w, 3));

       result.axis_iter_mut(Axis(0))
           .into_par_iter()
           .enumerate()
           .for_each(|(y, mut row)| {
               // Process each row in parallel
           });

       result
   }
   ```

3. Implement `get_filter_matrix` with histogram computation
4. Implement `precompute_filter_matrices` with linear interpolation

### Phase 3: Image Processing

**Deliverables:**
- Single image correction with EXIF preservation
- Preview generation
- Tauri commands for image operations

**Key Crates:**
- `kamadak-exif` or `rexif` - EXIF handling
- `base64` - Preview encoding for frontend

**Tasks:**

1. Implement `correct_image` in `src-tauri/src/core/processing/image.rs`:
   ```rust
   pub fn correct_image(
       input_path: &Path,
       output_path: &Path,
   ) -> Result<Vec<u8>, Error> {
       // Load image with EXIF
       // Apply correction
       // Save with EXIF preserved
       // Return preview bytes
   }
   ```

2. Create Tauri command in `src-tauri/src/commands/image.rs`:
   ```rust
   #[tauri::command]
   async fn correct_image(
       input: String,
       output: String,
       use_deep: bool,
   ) -> Result<String, String> {
       // Call core function
       // Return base64 preview
   }
   ```

### Phase 4: Video Processing with ez-ffmpeg

**Deliverables:**
- Video analysis with progress events
- Custom FrameFilter for color correction
- Video processing with frame streaming
- Tauri event emission for progress updates

**Key Crates:**
- `ez-ffmpeg` - Safe FFmpeg bindings with custom filter support
- `crossbeam-channel` - Progress communication

**Requirements:**
- Rust 1.80.0+
- FFmpeg 7.0+

**Why ez-ffmpeg:**
- Safe Rust API without unsafe code
- Custom `FrameFilter` trait for pixel-level processing
- Direct frame data access via `get_data_mut()`
- Optional GPU acceleration with OpenGL feature
- Broad codec support via FFmpeg

**Tasks:**

1. Add ez-ffmpeg dependency:
   ```toml
   [dependencies]
   ez-ffmpeg = { version = "0.4", features = ["static"] }
   ```

2. Implement custom `ColorCorrectionFilter`:
   ```rust
   use ez_ffmpeg::core::filter::frame_filter::{FrameFilter, FrameFilterContext};
   use ez_ffmpeg::core::frame::Frame;
   use ffmpeg_sys_next::{AVMediaType, AVPixelFormat};

   use crate::core::color::filter::{apply_filter, FilterMatrix};

   pub struct ColorCorrectionFilter {
       filter_matrices: Vec<FilterMatrix>,
       current_frame: usize,
   }

   impl ColorCorrectionFilter {
       pub fn new(filter_matrices: Vec<FilterMatrix>) -> Self {
           Self {
               filter_matrices,
               current_frame: 0,
           }
       }
   }

   impl FrameFilter for ColorCorrectionFilter {
       fn media_type(&self) -> AVMediaType {
           AVMediaType::AVMEDIA_TYPE_VIDEO
       }

       fn filter_frame(
           &mut self,
           mut frame: Frame,
           _ctx: &FrameFilterContext,
       ) -> Result<Option<Frame>, String> {
           // Get the interpolated filter for current frame
           let filter = &self.filter_matrices[self.current_frame];
           self.current_frame += 1;

           // Access frame planes (YUV or RGB depending on pixel format)
           match frame.format() {
               AVPixelFormat::AV_PIX_FMT_RGB24 => {
                   let data = frame.get_data_mut(0)
                       .ok_or("Failed to get RGB plane")?;

                   // Apply color correction filter to RGB data
                   apply_filter_rgb_inplace(data, frame.width(), frame.height(), filter);
               }
               AVPixelFormat::AV_PIX_FMT_YUV420P => {
                   // For YUV, we need to convert or apply in YUV space
                   let y_data = frame.get_data_mut(0).ok_or("Failed to get Y plane")?;
                   let u_data = frame.get_data_mut(1).ok_or("Failed to get U plane")?;
                   let v_data = frame.get_data_mut(2).ok_or("Failed to get V plane")?;

                   apply_filter_yuv_inplace(y_data, u_data, v_data, filter);
               }
               _ => return Err("Unsupported pixel format".to_string()),
           }

           Ok(Some(frame))
       }
   }
   ```

3. Implement video processing pipeline:
   ```rust
   use ez_ffmpeg::FfmpegContext;

   pub fn process_video(
       input_path: &str,
       output_path: &str,
       filter_matrices: Vec<FilterMatrix>,
   ) -> Result<(), Box<dyn std::error::Error>> {
       let color_filter = ColorCorrectionFilter::new(filter_matrices);

       FfmpegContext::builder()
           .input(input_path)
           .filter_desc("format=rgb24")  // Convert to RGB for our filter
           .frame_filter(color_filter)   // Apply custom color correction
           .output(output_path)
           .build()?
           .run()?;

       Ok(())
   }
   ```

4. Implement two-pass processing with Tauri events:
   ```rust
   #[tauri::command]
   async fn analyze_video(
       window: tauri::Window,
       input: String,
       output: String,
   ) -> Result<VideoData, String> {
       use ez_ffmpeg::FfmpegContext;

       // First pass: sample frames for filter matrix computation
       let mut filter_indices = Vec::new();
       let mut filter_matrices = Vec::new();
       let mut frame_count = 0;

       let ctx = FfmpegContext::builder()
           .input(&input)
           .frame_filter(AnalysisFilter::new(|frame_idx, frame| {
               frame_count = frame_idx;
               window.emit("analysis-progress", frame_idx).ok();

               // Sample every N frames
               if frame_idx % (fps * SAMPLE_SECONDS) == 0 {
                   let matrix = compute_filter_matrix(&frame);
                   filter_indices.push(frame_idx);
                   filter_matrices.push(matrix);
               }
           }))
           .build()
           .map_err(|e| e.to_string())?;

       ctx.run().map_err(|e| e.to_string())?;

       Ok(VideoData {
           input_path: input,
           output_path: output,
           frame_count,
           filter_indices,
           filter_matrices,
       })
   }

   #[tauri::command]
   async fn process_video(
       window: tauri::Window,
       video_data: VideoData,
   ) -> Result<(), String> {
       // Precompute interpolated matrices
       let interpolated = precompute_filter_matrices(
           video_data.frame_count,
           &video_data.filter_indices,
           &video_data.filter_matrices,
       );

       // Create filter with progress callback
       let filter = ColorCorrectionFilter::new(interpolated)
           .with_progress(|percent| {
               window.emit("process-progress", percent).ok();
           });

       FfmpegContext::builder()
           .input(&video_data.input_path)
           .filter_desc("format=rgb24")
           .frame_filter(filter)
           .output(&video_data.output_path)
           .build()
           .map_err(|e| e.to_string())?
           .run()
           .map_err(|e| e.to_string())?;

       Ok(())
   }
   ```

5. Optional: GPU-accelerated processing with OpenGL:
   ```toml
   [dependencies]
   ez-ffmpeg = { version = "0.4", features = ["static", "opengl"] }
   ```

   ```rust
   // Use GLSL shader for color correction (GPU-accelerated)
   FfmpegContext::builder()
       .input(input_path)
       .gl_filter(ColorCorrectionShader::new(filter_matrix))
       .output(output_path)
       .build()?
       .run()?;
   ```

### Phase 5: Deep Learning Integration

**Deliverables:**
- ONNX Runtime integration for Deep SESR model
- Model conversion from Keras to ONNX
- Inference pipeline

**Key Crates:**
- `ort` (ONNX Runtime) - Model inference
- `ndarray` - Tensor operations

**Tasks:**

1. Convert Keras model to ONNX:
   ```python
   import tf2onnx
   model = tf.keras.models.load_model("deep_sesr_2x_1d.keras")
   tf2onnx.convert.from_keras(model, output_path="deep_sesr.onnx")
   ```

2. Implement ONNX inference in Rust:
   ```rust
   use ort::{Session, SessionBuilder};

   pub struct DeepSESR {
       session: Session,
   }

   impl DeepSESR {
       pub fn new() -> Result<Self, Error> {
           let session = SessionBuilder::new()?
               .with_model_from_file("deep_sesr.onnx")?;
           Ok(Self { session })
       }

       pub fn enhance(&self, img: &Array3<u8>) -> Result<Array3<u8>, Error> {
           // Preprocess, run inference, postprocess
       }
   }
   ```

### Phase 6: Frontend Development

**Deliverables:**
- Modern UI with Vue/React + TailwindCSS
- File drag-and-drop
- Real-time preview
- Progress indicators

**Tasks:**

1. Design component structure:
   ```
   App
   ├── Header
   ├── FileSelector (drag-drop zone)
   ├── FileList (selected files)
   ├── PreviewPanel (before/after comparison)
   ├── OptionsPanel (deep learning toggle)
   └── ProcessButton + ProgressBar
   ```

2. Implement Tauri API bindings:
   ```typescript
   import { invoke } from '@tauri-apps/api/tauri';
   import { listen } from '@tauri-apps/api/event';

   async function correctImage(input: string, output: string) {
     const preview = await invoke<string>('correct_image', { input, output });
     return `data:image/png;base64,${preview}`;
   }

   // Listen for progress events
   await listen('process-progress', (event) => {
     updateProgress(event.payload.percent);
     updatePreview(event.payload.preview);
   });
   ```

3. Implement responsive layout with TailwindCSS
4. Add dark/light theme support

### Phase 7: Testing and Optimization

**Deliverables:**
- Unit tests for core algorithms
- Integration tests for Tauri commands
- Performance benchmarks
- Memory profiling

**Tasks:**

1. Unit tests with reference images:
   ```rust
   #[cfg(test)]
   mod tests {
       #[test]
       fn test_hue_shift_red() {
           let input = load_test_image("underwater.png");
           let result = hue_shift_red(&input, 45.0);
           assert_image_similar(&result, "expected_hue_shifted.png", 0.01);
       }
   }
   ```

2. Benchmark against Python implementation:
   ```rust
   use criterion::{criterion_group, Criterion};

   fn bench_apply_filter(c: &mut Criterion) {
       c.bench_function("apply_filter 1080p", |b| {
           let img = create_test_image(1920, 1080);
           let filter = create_test_filter();
           b.iter(|| apply_filter(&img, &filter))
       });
   }
   ```

3. Profile with `cargo flamegraph` and optimize hot paths

### Phase 8: Packaging and Distribution

**Deliverables:**
- Windows installer (.msi)
- macOS app bundle (.dmg)
- Linux packages (.deb, .AppImage)
- Auto-update support

**Tasks:**

1. Configure Tauri bundler in `tauri.conf.json`:
   ```json
   {
     "tauri": {
       "bundle": {
         "identifier": "com.divecolorcorrector.app",
         "icon": ["icons/icon.ico", "icons/icon.icns"],
         "targets": ["msi", "dmg", "deb", "appimage"]
       },
       "updater": {
         "active": true,
         "endpoints": ["https://releases.example.com/{{target}}/{{current_version}}"]
       }
     }
   }
   ```

2. Set up GitHub Actions for multi-platform builds
3. Create code signing certificates
4. Set up update server

## Dependency Mapping

| Python | Rust Equivalent | Notes |
|--------|-----------------|-------|
| `numpy` | `ndarray` | N-dimensional arrays |
| `opencv-python` (images) | `image` | Image I/O, resize, format conversion |
| `opencv-python` (video) | `ez-ffmpeg` | Video I/O with custom FrameFilter support |
| `pillow` | `image` + `kamadak-exif` | EXIF preservation |
| `tensorflow` | `ort` (ONNX Runtime) | Model inference |
| `PySimpleGUI` | Tauri + Vue/React | Desktop UI |

## ez-ffmpeg Integration Details

### Crate Features

| Feature | Purpose |
|---------|---------|
| `static` | Static linking of FFmpeg libraries (recommended for distribution) |
| `opengl` | GPU-accelerated OpenGL filters with GLSL shaders |
| `rtmp` | Embedded RTMP server for streaming |
| `async` | Async/await support for non-blocking operations |

### Frame Format Considerations

Most videos use YUV420P format. Options for color correction:

1. **Convert to RGB**: Use `filter_desc("format=rgb24")` before custom filter
   - Pro: Matches our existing RGB-based algorithm
   - Con: Extra conversion overhead

2. **Native YUV processing**: Implement color correction in YUV space
   - Pro: No conversion overhead
   - Con: Algorithm needs adaptation

3. **GPU shader**: Use OpenGL feature with GLSL fragment shader
   - Pro: Massive performance boost for 4K+ video
   - Con: Requires OpenGL context, more complex setup

### Example GLSL Color Correction Shader

```glsl
#version 330 core

uniform sampler2D inputTexture;
uniform mat4 colorMatrix;  // Our 20-element filter as 4x4 + offsets

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(inputTexture, texCoord);

    // Apply color transformation matrix
    vec3 corrected;
    corrected.r = dot(color.rgb, colorMatrix[0].rgb) + colorMatrix[0].a;
    corrected.g = color.g * colorMatrix[1].g + colorMatrix[1].a;
    corrected.b = color.b * colorMatrix[2].b + colorMatrix[2].a;

    fragColor = vec4(clamp(corrected, 0.0, 1.0), color.a);
}
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| FFmpeg 7.0+ requirement | Bundle FFmpeg with static linking or document install steps |
| ez-ffmpeg API stability | Pin version, monitor releases, contribute upstream if needed |
| ONNX model conversion issues | Validate model outputs match Python implementation |
| Cross-platform FFmpeg builds | Use `static` feature, test in CI on all platforms |
| GPU shader compatibility | Fallback to CPU path if OpenGL unavailable |

## Success Criteria

1. **Correctness**: Output images match Python implementation within acceptable tolerance
2. **Performance**: 2-5x faster than Python for image processing
3. **Binary Size**: < 50MB for base application (without ML model)
4. **Startup Time**: < 500ms to interactive UI
5. **Memory Usage**: < 500MB for 4K video processing
