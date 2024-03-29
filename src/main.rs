use std::io::Write;
use std::time::Instant;

use clap::Parser;

use image::DynamicImage;
use v4l::Fraction;
use yolov8_rs::{Args, YOLOv8};

use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;
use zune_jpeg::JpegDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut model = YOLOv8::new(args).unwrap();
    model.summary(); // model info

    let _ = opencv::core::set_num_threads(1);
    // Create a new capture device with a few extra parameters
    let mut webcam = Device::new(0).expect("Failed to open device");

    let width = 1280;
    let height = 720;
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
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"MJPG");
        webcam.set_format(&fmt).expect("Failed to write format");
    }

    let mut webcam_masked = Device::new(10).expect("Failed to open device");
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

        /*
        let mut fmt = webcam_masked.format().unwrap();
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"MJPG");
        let sink_fmt = webcam_masked
            .set_format(&fmt)
            .expect("Failed to write format");
        let source_fmt = webcam.format().unwrap();

        dbg!(source_fmt, sink_fmt);

        if source_fmt.width != sink_fmt.width || source_fmt.height != sink_fmt.height {
            panic!("failed to enforce source format on sink device");
        }

        let loopback_caps = webcam_masked.query_controls().unwrap();
        println!("Active out controls:\n{:?}", loopback_caps);

        println!(
            "Active out format:\n{}",
            v4l::video::Output::format(&webcam_masked).unwrap()
        );

        let mut loopback_caps = v4l::video::Output::params(&webcam_masked).unwrap();

        loopback_caps.interval = Fraction::new(1, 60);
        println!("Active out parameters:\n{loopback_caps}",);
        v4l::video::Output::set_params(&webcam_masked, &loopback_caps).unwrap();
        // v4l::video::Capture::set_params(&webcam_masked, &loopback_caps).unwrap();
        */
    }

    let buffer_count = 4;
    let mut in_stream = Stream::with_buffers(&webcam, Type::VideoCapture, buffer_count)
        .expect("Failed to create buffer stream");

    let mut out_stream =
        Stream::with_buffers(&webcam_masked, Type::VideoOutput, buffer_count).unwrap();
    let person_index = model
        .names()
        .iter()
        .enumerate()
        .find_map(|(idx, name)| if name == "person" { Some(idx) } else { None })
        .expect("Model missing `person` bbox name");

    loop {
        let (jpeg, buf_in_meta) = CaptureStream::next(&mut in_stream).unwrap();
        let (buf_out, buf_out_meta) = v4l::io::traits::OutputStream::next(&mut out_stream).unwrap();

        let start = Instant::now();

        use opencv::{core::*, imgproc::*};

        let mut decoder = JpegDecoder::new(jpeg);
        // decode the file
        let mut pixels = decoder.decode().unwrap();

        //let yuv = Mat::from_slice_rows_cols(buf, height, width as usize).unwrap();
        let rgb = unsafe {
            Mat::new_size_with_data(
                Size {
                    width: width as i32,
                    height: height as i32,
                },
                CV_8UC3,
                pixels.as_mut_ptr().cast(),
                Mat_AUTO_STEP,
            )
        }
        .unwrap();

        let mut rgba = unsafe {
            Mat::new_size(
                Size {
                    width: width as i32,
                    height: height as i32,
                },
                CV_8UC4,
            )
        }
        .unwrap();

        opencv::imgproc::cvt_color(&rgb, &mut rgba, COLOR_RGB2RGBA, 0).unwrap();

        let len = unsafe { rgba.dataend().offset_from(rgba.datastart()) } as usize;
        let rgba_data = unsafe { std::slice::from_raw_parts(rgba.datastart(), len) }.to_owned();
        let rgba8 = image::RgbaImage::from_raw(width, height, rgba_data).unwrap();

        let img = DynamicImage::ImageRgba8(rgba8);
        let ys = model.run(&[img]).unwrap();
        if let Some(ys) = ys.first() {
            let detection_names: Vec<_> = ys
                .bboxes
                .iter()
                .map(|b| model.names()[b.id()].as_str())
                .collect();

            let Some(person_index) = detection_names.iter().position(|name| *name == "person")
            else {
                println!("No person found");
                continue;
            };

            if let (Some(mask), Some(bbox)) =
                (ys.masks.get(person_index), ys.bboxes.get(person_index))
            {
                let mut pixels = Vec::new();
                for p in mask.pixels() {
                    pixels.push(p.0);
                }

                let greyscale = unsafe {
                    Mat::new_size_with_data(
                        Size {
                            width: width as i32,
                            height: height as i32,
                        },
                        CV_8UC1,
                        pixels.as_mut_ptr().cast(),
                        Mat_AUTO_STEP,
                    )
                }
                .unwrap();

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

                opencv::imgproc::cvt_color(&greyscale, &mut mask_rgba, COLOR_GRAY2RGBA, 0).unwrap();

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
                opencv::core::multiply(&mask_rgba, &rgba, &mut masked, 1.0 / 255.0, -1).unwrap();
                let len = unsafe { masked.dataend().offset_from(masked.datastart()) } as usize;
                let masked_data = unsafe { std::slice::from_raw_parts(masked.datastart(), len) };

                let masked_image =
                    image::RgbaImage::from_raw(width, height, masked_data.to_owned()).unwrap();
                let masked_jpeg =
                    turbojpeg::compress_image(&masked_image, 85, turbojpeg::Subsamp::Sub2x2)
                        .unwrap();

                let buf_out = &mut buf_out[..masked_jpeg.len()];
                buf_out.copy_from_slice(&masked_jpeg);
                buf_out_meta.field = 0;
                buf_out_meta.bytesused = masked_jpeg.len() as u32;

                // println!("Buffer");
                // println!("  sequence   [in] : {}", buf_in_meta.sequence);
                // println!("  sequence  [out] : {}", buf_out_meta.sequence);
                // println!("  timestamp  [in] : {}", buf_in_meta.timestamp);
                // println!("  timestamp [out] : {}", buf_out_meta.timestamp);
                // println!("  flags      [in] : {}", buf_in_meta.flags);
                // println!("  flags     [out] : {}", buf_out_meta.flags);
                // println!("  length     [in] : {}", jpeg.len());
                // println!("  length    [out] : {}", buf_out.len());
            }
        }

        println!("Observed latency: {:?}", start.elapsed());
    }

    Ok(())
}
