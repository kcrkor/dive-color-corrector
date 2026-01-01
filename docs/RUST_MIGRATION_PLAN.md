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

### Phase 4: Video Processing

**Deliverables:**
- Video analysis with progress events
- Video processing with frame streaming
- Tauri event emission for progress updates

**Key Crates:**
- `opencv` (via `opencv-rust`) or `ffmpeg-next` - Video I/O
- `crossbeam-channel` - Progress communication

**Tasks:**

1. Implement video analysis with Tauri events:
   ```rust
   #[tauri::command]
   async fn analyze_video(
       window: tauri::Window,
       input: String,
       output: String,
   ) -> Result<VideoData, String> {
       // Open video
       // Sample frames, emit progress
       window.emit("analysis-progress", frame_count)?;
       // Return video data
   }
   ```

2. Implement video processing with streaming:
   ```rust
   #[tauri::command]
   async fn process_video(
       window: tauri::Window,
       video_data: VideoData,
       yield_preview: bool,
   ) -> Result<(), String> {
       // Precompute filter matrices
       // Process frames with progress events
       window.emit("process-progress", ProcessProgress {
           percent,
           preview: preview_base64,
       })?;
   }
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
| `opencv-python` | `opencv-rust` or `image` | Image I/O, resize |
| `pillow` | `image` + `kamadak-exif` | EXIF preservation |
| `tensorflow` | `ort` (ONNX Runtime) | Model inference |
| `PySimpleGUI` | Tauri + Vue/React | Desktop UI |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OpenCV Rust bindings complexity | Consider pure-Rust `image` crate for basic operations |
| ONNX model conversion issues | Validate model outputs match Python implementation |
| Video codec support | Use FFmpeg via `ffmpeg-next` for broad format support |
| Cross-platform builds | Extensive CI testing on all target platforms |

## Success Criteria

1. **Correctness**: Output images match Python implementation within acceptable tolerance
2. **Performance**: 2-5x faster than Python for image processing
3. **Binary Size**: < 50MB for base application (without ML model)
4. **Startup Time**: < 500ms to interactive UI
5. **Memory Usage**: < 500MB for 4K video processing
