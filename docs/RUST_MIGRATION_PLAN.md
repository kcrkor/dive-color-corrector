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

### Phase 4: Video Processing with FFmpeg + Rust Filter

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                        FFmpeg                                │
│  ┌──────────┐   ┌──────────┐       ┌──────────┐   ┌───────┐ │
│  │  Demux   │ → │  Decode  │ → ... │  Encode  │ → │  Mux  │ │
│  └──────────┘   └──────────┘   ↑   └──────────┘   └───────┘ │
└────────────────────────────────┼────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Rust Color Filter     │
                    │  (Pure Rust, no FFI)    │
                    │  - apply_filter()       │
                    │  - hue_shift_red()      │
                    │  - precompute_matrices()│
                    └─────────────────────────┘
```

**Deliverables:**
- FFmpeg-based video decode/encode pipeline
- Pure Rust color correction filter (ported from Python)
- Two-pass processing: analyze → precompute → apply
- Tauri event emission for progress updates

**Key Crates:**
- `ffmpeg-next` - FFmpeg bindings for video I/O
- `ndarray` - Frame data manipulation
- `rayon` - Parallel pixel processing
- `crossbeam-channel` - Progress communication

**Tasks:**

1. Add dependencies:
   ```toml
   [dependencies]
   ffmpeg-next = "7.0"
   ndarray = "0.15"
   rayon = "1.8"
   ```

2. Implement video decoder/encoder wrapper:
   ```rust
   use ffmpeg_next as ffmpeg;
   use ffmpeg::{codec, format, frame, media, software::scaling};
   use ndarray::Array3;

   pub struct VideoProcessor {
       input_ctx: format::context::Input,
       output_ctx: format::context::Output,
       decoder: codec::decoder::Video,
       encoder: codec::encoder::Video,
       scaler: scaling::Context,
       video_stream_index: usize,
       frame_count: usize,
       fps: f64,
   }

   impl VideoProcessor {
       pub fn new(input_path: &str, output_path: &str) -> Result<Self, ffmpeg::Error> {
           ffmpeg::init()?;

           let input_ctx = format::input(input_path)?;
           let input_stream = input_ctx
               .streams()
               .best(media::Type::Video)
               .ok_or(ffmpeg::Error::StreamNotFound)?;

           let video_stream_index = input_stream.index();
           let decoder = ffmpeg::codec::context::Context::from_parameters(
               input_stream.parameters()
           )?.decoder().video()?;

           let fps = input_stream.avg_frame_rate().into();
           let frame_count = input_stream.frames() as usize;

           // Set up output
           let mut output_ctx = format::output(output_path)?;
           let codec = ffmpeg::encoder::find(codec::Id::H264)
               .ok_or(ffmpeg::Error::EncoderNotFound)?;

           let mut output_stream = output_ctx.add_stream(codec)?;
           let mut encoder = codec::context::Context::new_with_codec(codec)
               .encoder().video()?;

           encoder.set_width(decoder.width());
           encoder.set_height(decoder.height());
           encoder.set_format(format::Pixel::YUV420P);
           encoder.set_time_base(input_stream.time_base());

           output_stream.set_parameters(&encoder);

           // Scaler: convert decoded frame to RGB for processing
           let scaler = scaling::Context::get(
               decoder.format(), decoder.width(), decoder.height(),
               format::Pixel::RGB24, decoder.width(), decoder.height(),
               scaling::Flags::BILINEAR,
           )?;

           Ok(Self {
               input_ctx, output_ctx, decoder, encoder, scaler,
               video_stream_index, frame_count, fps,
           })
       }

       pub fn frame_count(&self) -> usize { self.frame_count }
       pub fn fps(&self) -> f64 { self.fps }
   }
   ```

3. Implement frame iteration and processing:
   ```rust
   impl VideoProcessor {
       /// Decode frames and yield RGB data for processing
       pub fn decode_frames(&mut self) -> impl Iterator<Item = (usize, Array3<u8>)> + '_ {
           let mut frame_idx = 0;
           let mut decoded = frame::Video::empty();
           let mut rgb_frame = frame::Video::empty();

           self.input_ctx.packets()
               .filter_map(move |(stream, packet)| {
                   if stream.index() != self.video_stream_index {
                       return None;
                   }

                   self.decoder.send_packet(&packet).ok()?;

                   while self.decoder.receive_frame(&mut decoded).is_ok() {
                       // Convert to RGB
                       self.scaler.run(&decoded, &mut rgb_frame).ok()?;

                       // Convert to ndarray
                       let width = rgb_frame.width() as usize;
                       let height = rgb_frame.height() as usize;
                       let data = rgb_frame.data(0);
                       let stride = rgb_frame.stride(0);

                       let mut arr = Array3::<u8>::zeros((height, width, 3));
                       for y in 0..height {
                           let row_start = y * stride;
                           for x in 0..width {
                               let px = row_start + x * 3;
                               arr[[y, x, 0]] = data[px];     // R
                               arr[[y, x, 1]] = data[px + 1]; // G
                               arr[[y, x, 2]] = data[px + 2]; // B
                           }
                       }

                       frame_idx += 1;
                       return Some((frame_idx, arr));
                   }
                   None
               })
       }

       /// Encode processed RGB frame back to video
       pub fn encode_frame(&mut self, rgb_data: &Array3<u8>) -> Result<(), ffmpeg::Error> {
           // Convert RGB back to YUV420P for encoding
           // ... encoding logic
           Ok(())
       }
   }
   ```

4. Implement pure Rust color correction filter:
   ```rust
   use ndarray::{Array3, Axis, s};
   use rayon::prelude::*;

   pub type FilterMatrix = [f32; 20];

   /// Apply color correction filter to RGB frame (in-place, parallelized)
   pub fn apply_filter(frame: &mut Array3<u8>, filter: &FilterMatrix) {
       let (height, width, _) = frame.dim();

       // Process rows in parallel
       frame.axis_iter_mut(Axis(0))
           .into_par_iter()
           .for_each(|mut row| {
               for x in 0..width {
                   let r = row[[x, 0]] as f32;
                   let g = row[[x, 1]] as f32;
                   let b = row[[x, 2]] as f32;

                   // Apply filter matrix
                   let new_r = r * filter[0] + g * filter[1] + b * filter[2] + filter[4] * 255.0;
                   let new_g = g * filter[6] + filter[9] * 255.0;
                   let new_b = b * filter[12] + filter[14] * 255.0;

                   row[[x, 0]] = new_r.clamp(0.0, 255.0) as u8;
                   row[[x, 1]] = new_g.clamp(0.0, 255.0) as u8;
                   row[[x, 2]] = new_b.clamp(0.0, 255.0) as u8;
               }
           });
   }

   /// Compute filter matrix from frame (analysis pass)
   pub fn get_filter_matrix(frame: &Array3<u8>) -> FilterMatrix {
       // Resize to 256x256 for analysis
       let resized = resize_frame(frame, 256, 256);

       // Calculate average RGB
       let (avg_r, avg_g, avg_b) = calculate_mean_rgb(&resized);

       // Find hue shift for red channel
       let hue_shift = find_hue_shift(avg_r, MIN_AVG_RED, MAX_HUE_SHIFT);

       // Apply hue shift and compute histograms
       let shifted = apply_hue_shift(&resized, hue_shift);
       let (hist_r, hist_g, hist_b) = compute_histograms(&shifted);

       // Find normalizing intervals
       let threshold = (256 * 256) as f32 / THRESHOLD_RATIO;
       let (r_low, r_high) = find_normalizing_interval(&hist_r, threshold);
       let (g_low, g_high) = find_normalizing_interval(&hist_g, threshold);
       let (b_low, b_high) = find_normalizing_interval(&hist_b, threshold);

       // Compute gains and offsets
       let (shifted_r, shifted_g, shifted_b) = hue_shift_coefficients(hue_shift);

       let red_gain = 256.0 / (r_high - r_low);
       let green_gain = 256.0 / (g_high - g_low);
       let blue_gain = 256.0 / (b_high - b_low);

       let red_offset = (-r_low / 256.0) * red_gain;
       let green_offset = (-g_low / 256.0) * green_gain;
       let blue_offset = (-b_low / 256.0) * blue_gain;

       [
           shifted_r * red_gain,
           shifted_g * red_gain,
           shifted_b * red_gain * BLUE_MAGIC_VALUE,
           0.0, red_offset,
           0.0, green_gain, 0.0, 0.0, green_offset,
           0.0, 0.0, blue_gain, 0.0, blue_offset,
           0.0, 0.0, 0.0, 1.0, 0.0,
       ]
   }

   /// Precompute interpolated filter matrices for all frames
   pub fn precompute_filter_matrices(
       frame_count: usize,
       indices: &[usize],
       matrices: &[FilterMatrix],
   ) -> Vec<FilterMatrix> {
       (0..frame_count)
           .map(|frame_idx| {
               // Linear interpolation between sampled matrices
               interpolate_filter(frame_idx, indices, matrices)
           })
           .collect()
   }
   ```

5. Implement two-pass video processing with Tauri:
   ```rust
   #[tauri::command]
   async fn analyze_video(
       window: tauri::Window,
       input: String,
       output: String,
   ) -> Result<VideoData, String> {
       let mut processor = VideoProcessor::new(&input, &output)
           .map_err(|e| e.to_string())?;

       let fps = processor.fps();
       let sample_interval = (fps * SAMPLE_SECONDS as f64) as usize;

       let mut filter_indices = Vec::new();
       let mut filter_matrices = Vec::new();
       let mut frame_count = 0;

       // Analysis pass: sample frames for filter computation
       for (idx, frame) in processor.decode_frames() {
           frame_count = idx;

           // Emit progress
           window.emit("analysis-progress", idx).ok();

           // Sample every N frames
           if idx % sample_interval == 0 {
               let matrix = get_filter_matrix(&frame);
               filter_indices.push(idx);
               filter_matrices.push(matrix);
           }
       }

       Ok(VideoData {
           input_path: input,
           output_path: output,
           frame_count,
           fps,
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
       let matrices = precompute_filter_matrices(
           video_data.frame_count,
           &video_data.filter_indices,
           &video_data.filter_matrices,
       );

       let mut processor = VideoProcessor::new(
           &video_data.input_path,
           &video_data.output_path,
       ).map_err(|e| e.to_string())?;

       // Processing pass: apply filters
       for (idx, mut frame) in processor.decode_frames() {
           // Apply color correction (pure Rust)
           apply_filter(&mut frame, &matrices[idx - 1]);

           // Encode back to video (FFmpeg)
           processor.encode_frame(&frame).map_err(|e| e.to_string())?;

           // Emit progress
           let percent = (idx as f64 / video_data.frame_count as f64) * 100.0;
           window.emit("process-progress", percent).ok();
       }

       processor.finalize().map_err(|e| e.to_string())?;
       Ok(())
   }
   ```

6. Implement hue shift algorithm in Rust:
   ```rust
   use std::f32::consts::PI;

   /// Hue shift transformation for red channel recovery
   pub fn hue_shift_red(rgb: &Array3<f32>, hue_degrees: f32) -> Array3<f32> {
       let u = (hue_degrees * PI / 180.0).cos();
       let w = (hue_degrees * PI / 180.0).sin();

       let (height, width, _) = rgb.dim();
       let mut result = Array3::<f32>::zeros((height, width, 3));

       // Transformation coefficients
       let r_from_r = 0.299 + 0.701 * u + 0.168 * w;
       let r_from_g = 0.587 - 0.587 * u + 0.330 * w;
       let r_from_b = 0.114 - 0.114 * u - 0.497 * w;

       ndarray::Zip::from(result.rows_mut())
           .and(rgb.rows())
           .par_for_each(|mut out_row, in_row| {
               for x in 0..width {
                   let r = in_row[[x, 0]];
                   let g = in_row[[x, 1]];
                   let b = in_row[[x, 2]];

                   out_row[[x, 0]] = r * r_from_r + g * r_from_g + b * r_from_b;
                   out_row[[x, 1]] = g;
                   out_row[[x, 2]] = b;
               }
           });

       result
   }
   ```

### Phase 5: Deep Learning Integration

**Model Analysis (Deep SESR):**

| Property | Value |
|----------|-------|
| Format | Keras 3.9.0 (.keras) |
| Size | 9.9 MB (weights: 10.2 MB) |
| Input | (batch, 240, 320, 3) float32, normalized [0, 1] |
| Output | (batch, 240, 320, 3) float32, range [-1, 1] |
| Layers | 150 total |

**Layer Composition:**
- Conv2D: 48 layers
- Concatenate: 35 (dense connections)
- Activation: 28 (relu, sigmoid, tanh, linear)
- BatchNormalization: 27
- Add: 10 (residual connections)
- UpSampling2D: 1

**ONNX Compatibility: ✅ All layers fully supported**

**Deliverables:**
- ONNX Runtime integration for Deep SESR model
- Model conversion from Keras 3.x to ONNX
- Inference pipeline with pre/post processing

**Key Crates:**
- `ort` (ONNX Runtime) - Model inference
- `ndarray` - Tensor operations
- `image` - Image resizing for pre/post processing

**Tasks:**

1. Convert Keras model to ONNX:
   ```python
   import tensorflow as tf
   import tf2onnx

   # Load Keras 3.x model
   model = tf.keras.models.load_model("deep_sesr_2x_1d.keras")

   # Convert to ONNX with opset 17 for best compatibility
   spec = (tf.TensorSpec((1, 240, 320, 3), tf.float32, name="input"),)
   output_path = "deep_sesr.onnx"

   model_proto, _ = tf2onnx.convert.from_keras(
       model,
       input_signature=spec,
       opset=17,
       output_path=output_path
   )

   print(f"Model saved to {output_path}")
   print(f"Inputs: {[i.name for i in model_proto.graph.input]}")
   print(f"Outputs: {[o.name for o in model_proto.graph.output]}")
   ```

2. Validate ONNX model outputs match TensorFlow:
   ```python
   import onnxruntime as ort
   import numpy as np

   # Load both models
   keras_model = tf.keras.models.load_model("deep_sesr_2x_1d.keras")
   onnx_session = ort.InferenceSession("deep_sesr.onnx")

   # Test with random input
   test_input = np.random.rand(1, 240, 320, 3).astype(np.float32)

   keras_output = keras_model.predict(test_input)
   onnx_output = onnx_session.run(None, {"input": test_input})[0]

   # Check outputs match within tolerance
   assert np.allclose(keras_output, onnx_output, rtol=1e-5, atol=1e-5)
   ```

3. Implement ONNX inference in Rust:
   ```rust
   use ndarray::{Array3, Array4, s};
   use ort::{Session, Value, inputs};
   use image::imageops::FilterType;

   pub struct DeepSESR {
       session: Session,
       input_height: u32,
       input_width: u32,
   }

   impl DeepSESR {
       pub fn new(model_path: &str) -> Result<Self, ort::Error> {
           ort::init().commit()?;

           let session = Session::builder()?
               .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
               .with_intra_threads(4)?
               .commit_from_file(model_path)?;

           Ok(Self {
               session,
               input_height: 240,
               input_width: 320,
           })
       }

       pub fn enhance(&self, img: &Array3<u8>) -> Result<Array3<u8>, ort::Error> {
           let (orig_h, orig_w, _) = img.dim();

           // Preprocess: resize and normalize to [0, 1]
           let resized = resize_image(img, self.input_width, self.input_height);
           let normalized = resized.mapv(|x| x as f32 / 255.0);

           // Add batch dimension: (H, W, 3) -> (1, H, W, 3)
           let input = normalized.insert_axis(ndarray::Axis(0));

           // Run inference
           let outputs = self.session.run(
               inputs!["input" => input.view()]?
           )?;

           // Get output tensor
           let output: ndarray::ArrayView4<f32> = outputs[0]
               .try_extract_tensor()?
               .view()
               .into_dimensionality()?;

           // Postprocess: remove batch, scale [-1,1] -> [0,255]
           let enhanced = output
               .index_axis(ndarray::Axis(0), 0)
               .mapv(|x| ((x + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8);

           // Resize back to original dimensions
           let result = resize_image(&enhanced, orig_w as u32, orig_h as u32);

           Ok(result)
       }
   }

   fn resize_image(img: &Array3<u8>, width: u32, height: u32) -> Array3<u8> {
       // Use image crate for high-quality resizing
       // ... implementation
   }
   ```

4. Bundle ONNX model with Tauri app:
   ```json
   // tauri.conf.json
   {
     "tauri": {
       "bundle": {
         "resources": ["models/deep_sesr.onnx"]
       }
     }
   }
   ```

5. Optional: Quantize model for smaller size and faster inference:
   ```python
   from onnxruntime.quantization import quantize_dynamic, QuantType

   quantize_dynamic(
       "deep_sesr.onnx",
       "deep_sesr_int8.onnx",
       weight_type=QuantType.QUInt8
   )
   # Reduces ~10MB -> ~3MB with minimal quality loss
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
| `numpy` | `ndarray` | N-dimensional arrays with parallel iteration |
| `opencv-python` (images) | `image` | Image I/O, resize, format conversion |
| `opencv-python` (video) | `ffmpeg-next` | FFmpeg bindings for video decode/encode |
| `pillow` | `image` + `kamadak-exif` | EXIF preservation |
| `tensorflow` | `ort` (ONNX Runtime) | Model inference |
| `PySimpleGUI` | Tauri + Vue/React | Desktop UI |

## FFmpeg Integration Architecture

### Separation of Concerns

```
┌────────────────────────────────────────────────────────────────┐
│                    ffmpeg-next (Rust bindings)                  │
├────────────────────────────────────────────────────────────────┤
│  Demuxing    │  Decoding    │  Scaling     │  Encoding  │ Mux  │
│  (container) │  (H264/VP9)  │  (YUV→RGB)   │  (H264)    │      │
└──────────────┴──────────────┴──────┬───────┴────────────┴──────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │      Pure Rust Processing       │
                    ├─────────────────────────────────┤
                    │  • apply_filter() - rayon      │
                    │  • get_filter_matrix()          │
                    │  • hue_shift_red()              │
                    │  • precompute_filter_matrices() │
                    │  • histogram computation        │
                    └─────────────────────────────────┘
```

### Why This Approach

1. **FFmpeg handles complexity**: Container formats, codecs, pixel format conversion
2. **Rust handles performance**: Parallel pixel processing with rayon
3. **Clean boundary**: RGB frames passed between FFmpeg and Rust filter
4. **Testable**: Filter functions can be unit tested without video I/O

### Frame Processing Pipeline

```
Video File → Demux → Decode → Scale(YUV→RGB) → [Rust Filter] → Scale(RGB→YUV) → Encode → Mux → Output
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │   apply_filter(&mut frame,    │
                                    │                &filter_matrix) │
                                    │   - Parallel row processing   │
                                    │   - In-place modification     │
                                    │   - SIMD-friendly layout      │
                                    └───────────────────────────────┘
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| FFmpeg linking complexity | Use system FFmpeg or bundle with static linking |
| ffmpeg-next API changes | Pin version, wrap in abstraction layer |
| ONNX model conversion issues | Validate model outputs match Python implementation |
| Cross-platform FFmpeg builds | Test in CI on Windows/macOS/Linux |
| Performance regression | Benchmark against Python, optimize hot paths with rayon |

## Success Criteria

1. **Correctness**: Output images match Python implementation within acceptable tolerance
2. **Performance**: 2-5x faster than Python for image processing
3. **Binary Size**: < 50MB for base application (without ML model)
4. **Startup Time**: < 500ms to interactive UI
5. **Memory Usage**: < 500MB for 4K video processing
