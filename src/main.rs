use std::time::Instant;

use clap::Parser;

use crossbeam_channel::Receiver;
use opencv::core::{Size, CV_8UC4};
use opencv::prelude::*;

use image::DynamicImage;
use turbojpeg::Compressor;
use turbojpeg::Decompressor;
use turbojpeg::OutputBuf;
use yolov8_rs::{Args, YOLOv8};

use v4l::buffer::{Metadata, Type};
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = opencv::core::set_num_threads(1);
    let mut args = Args::parse();
    //args.profile = true;

    let model = YOLOv8::new(args).unwrap();
    model.summary(); // model info

    // ========== Create Input Device ==========

    // Create a new capture device with a few extra parameters
    let webcam = Device::new(0).expect("Failed to open device");

    let width = 1280usize;
    let height = 720usize;
    {
        let format = webcam.format().unwrap();
        println!("Active input format:\n{}", format);

        let params = webcam.params().unwrap();
        println!("Active input parameters:\n{}", params);

        println!("Available input formats:");
        for format in webcam.enum_formats().unwrap() {
            println!("  {} ({})", format.fourcc, format.description);

            for framesize in webcam.enum_framesizes(format.fourcc).unwrap() {
                for discrete in framesize.size.to_discrete() {
                    println!("    Size: {}", discrete);

                    for frameinterval in webcam
                        .enum_frameintervals(framesize.fourcc, discrete.width, discrete.height)
                        .unwrap()
                    {
                        println!("      Interval:  {}", frameinterval);
                    }
                }
            }

            println!()
        }

        // Let's say we want to explicitly request another format
        let mut fmt = webcam.format().expect("Failed to read format");
        fmt.width = width as u32;
        fmt.height = height as u32;
        fmt.fourcc = FourCC::new(b"MJPG");
        webcam.set_format(&fmt).expect("Failed to write format");
    }

    // ========== Create Output Device ==========

    let webcam_masked = Device::new(10).expect("Failed to open device");
    {
        let source_fmt = Capture::format(&webcam)?;
        let sink_fmt = v4l::video::Output::set_format(&webcam_masked, &source_fmt)?;
        if source_fmt.width != sink_fmt.width
            || source_fmt.height != sink_fmt.height
            || source_fmt.fourcc != sink_fmt.fourcc
        {
            panic!("failed to enforce source format on sink device");
        }
        println!(
            "New out format:\n{}",
            v4l::video::Output::format(&webcam_masked)?
        );
    }

    // ========== Create Buffers ==========

    let buffer_count = 4;
    let mut in_stream = Stream::with_buffers(&webcam, Type::VideoCapture, buffer_count)
        .expect("Failed to create buffer stream");

    let (tx, rx) = crossbeam_channel::bounded(4);
    let _process_task = process(rx, model, webcam_masked, width, height);

    loop {
        let (jpeg, buf_in_meta) = CaptureStream::next(&mut in_stream).unwrap();

        if tx.send((jpeg.to_owned(), buf_in_meta.clone())).is_err() {
            println!("Process buffer full, dropping frame!");
        }
    }
}

fn process(
    rx: Receiver<(Vec<u8>, Metadata)>,
    mut model: YOLOv8,
    webcam_masked: Device,
    width: usize,
    height: usize,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let person_index = model
            .names()
            .iter()
            .enumerate()
            .find_map(|(idx, name)| if name == "person" { Some(idx) } else { None })
            .expect("Model missing `person` bbox name");
        // ========== General Allocations ==========
        let mut out_stream = Stream::with_buffers(&webcam_masked, Type::VideoOutput, 4).unwrap();

        let mut compressor = Compressor::new().unwrap();
        let mut decompressor = Decompressor::new().unwrap();
        let mut jpeg_buf = OutputBuf::new_owned();
        let mut rgba_pixels = vec![0; 4 * (width * height) as usize];

        // SAFETY: Memory allocated by opencv
        let mut mask_rgba = unsafe {
            Mat::new_size(
                Size {
                    width: width as i32,
                    height: height as i32,
                },
                CV_8UC4,
            )
        }
        .unwrap();

        // SAFETY: Memory allocated by opencv
        let mut masked = unsafe {
            Mat::new_size(
                Size {
                    width: width as i32,
                    height: height as i32,
                },
                CV_8UC4,
            )
        }
        .unwrap();

        // ========== Process Frame Loop ==========
        loop {
            let (jpeg, buf_in_meta) = rx.recv().unwrap();

            let (buf_out, buf_out_meta) =
                v4l::io::traits::OutputStream::next(&mut out_stream).unwrap();

            let iteration_start = Instant::now();

            use opencv::{core::*, imgproc::*};

            let start = Instant::now();
            decompressor
                .decompress(
                    &jpeg,
                    turbojpeg::Image {
                        pixels: rgba_pixels.as_mut_slice(),
                        width: width as usize,
                        pitch: 4 * width as usize,
                        height: height as usize,
                        format: turbojpeg::PixelFormat::RGBA,
                    },
                )
                .unwrap();
            //println!("Decompress took: {:?}", start.elapsed());

            // SAFETY:
            // 1. `rgb_pixels` lives for the entire duration of the loop
            // 2. `rgb_pixels` has len `width * height * 3` by allocation above
            // 3. `rgb_pixels` decoded by turbojpeg as turbojpeg::PixelFormat::RGB
            let rgba = unsafe {
                Mat::new_size_with_data(
                    Size {
                        width: width as i32,
                        height: height as i32,
                    },
                    CV_8UC4,
                    rgba_pixels.as_mut_ptr().cast(),
                    Mat_AUTO_STEP,
                )
            }
            .unwrap();

            //println!("Get rgba source image took: {:?}", start.elapsed());

            // # SAFETY:
            // `rgba.datastart()` and `rgba.dataend()` come from the same allocation
            let len = unsafe { rgba.dataend().offset_from(rgba.datastart()) } as usize;

            // # SAFETY:
            // Buffer of `len` bytes allocated by opencv
            let rgba_data = unsafe { std::slice::from_raw_parts(rgba.datastart(), len) }.to_owned();
            let rgba8 = image::RgbaImage::from_raw(width as u32, height as u32, rgba_data).unwrap();

            let img = DynamicImage::ImageRgba8(rgba8);
            let start = Instant::now();
            let ys = model.run(&img).unwrap();
            //println!("Model eval took: {:?}", start.elapsed());

            let Some(mut ys) = ys else {
                continue;
            };
            let Some(person_index) = ys.bboxes.iter().position(|bb| bb.id == person_index) else {
                //println!("No person found");
                continue;
            };

            let Some(mask) = ys.masks.get_mut(person_index) else {
                continue;
            };

            let mut mask_pixels = mask.as_flat_samples_mut();
            let mask_pixels = mask_pixels.as_mut_slice();

            assert_eq!(width * height, mask_pixels.len().try_into().unwrap());
            // SAFETY:
            // By assert above, pixels is readable for `width * height` bytes, since it has 8
            // bits per channel
            let greyscale = unsafe {
                Mat::new_size_with_data(
                    Size {
                        width: width as i32,
                        height: height as i32,
                    },
                    CV_8UC1,
                    mask_pixels.as_mut_ptr().cast(),
                    Mat_AUTO_STEP,
                )
            }
            .unwrap();

            let start = Instant::now();
            opencv::imgproc::cvt_color(&greyscale, &mut mask_rgba, COLOR_GRAY2RGBA, 0).unwrap();
            //println!("cvt grayscale -> rgba took: {:?}", start.elapsed());

            let start = Instant::now();
            opencv::core::multiply(&mask_rgba, &rgba, &mut masked, 1.0 / 255.0, -1).unwrap();
            //println!("rgba, mask mutiply took: {:?}", start.elapsed());

            // # SAFETY:
            // `rgba.datastart()` and `rgba.dataend()` come from the same allocation
            let len = unsafe { masked.dataend().offset_from(masked.datastart()) } as usize;

            // # SAFETY:
            // Buffer of `len` bytes allocated by opencv
            let masked_data = unsafe { std::slice::from_raw_parts(masked.datastart(), len) };

            let start = Instant::now();
            compressor
                .compress(
                    turbojpeg::Image {
                        pixels: masked_data,
                        width: width as usize,
                        pitch: 4 * width as usize,
                        height: height as usize,
                        format: turbojpeg::PixelFormat::RGBA,
                    },
                    &mut jpeg_buf,
                )
                .unwrap();
            //println!("jpeg compress took: {:?}", start.elapsed());

            let start = Instant::now();
            let buf_out = &mut buf_out[..jpeg_buf.len()];
            buf_out.copy_from_slice(&jpeg_buf);
            buf_out_meta.field = 0;
            buf_out_meta.bytesused = jpeg_buf.len() as u32;
            //println!("Copy into v4l output buffer took: {:?}", start.elapsed());

            // println!("Buffer");
            // println!("  sequence   [in] : {}", buf_in_meta.sequence);
            // println!("  sequence  [out] : {}", buf_out_meta.sequence);
            // println!("  timestamp  [in] : {}", buf_in_meta.timestamp);
            // println!("  timestamp [out] : {}", buf_out_meta.timestamp);
            // println!("  flags      [in] : {}", buf_in_meta.flags);
            // println!("  flags     [out] : {}", buf_out_meta.flags);
            // println!("  length     [in] : {}", jpeg.len());
            // println!("  length    [out] : {}", buf_out.len());

            //println!("Full loop latency latency: {:?}", iteration_start.elapsed());
        }
    })
}
