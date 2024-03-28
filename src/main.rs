use std::time::Instant;

use clap::Parser;

use image::ColorType;
use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
use yolov8_rs::{Args, YOLOv8};

use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;
use zune_jpeg::JpegDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut model = YOLOv8::new(args)?;
    model.summary(); // model info

    opencv::core::set_num_threads(1);
    // Create a new capture device with a few extra parameters
    let mut dev = Device::new(0).expect("Failed to open device");

    let format = dev.format().unwrap();
    println!("Active format:\n{}", format);

    let params = dev.params().unwrap();
    println!("Active parameters:\n{}", params);

    println!("Available formats:");
    for format in dev.enum_formats().unwrap() {
        println!("  {} ({})", format.fourcc, format.description);

        for framesize in dev.enum_framesizes(format.fourcc).unwrap() {
            for discrete in framesize.size.to_discrete() {
                println!("    Size: {}", discrete);

                for frameinterval in dev
                    .enum_frameintervals(framesize.fourcc, discrete.width, discrete.height)
                    .unwrap()
                {
                    println!("      Interval:  {}", frameinterval);
                }
            }
        }

        println!()
    }

    let width = 1280;
    let height = 720;

    // Let's say we want to explicitly request another format
    let mut fmt = dev.format().expect("Failed to read format");
    fmt.width = width;
    fmt.height = height;
    fmt.fourcc = FourCC::new(b"MJPG");
    let fmt = dev.set_format(&fmt).expect("Failed to write format");

    println!("Format in use:\n{}", fmt);

    let mut stream = Stream::with_buffers(&mut dev, Type::VideoCapture, 4)
        .expect("Failed to create buffer stream");

    let win_name = "webcam";
    opencv::highgui::named_window(win_name, opencv::highgui::WINDOW_AUTOSIZE);

    let mut i = 0;
    loop {
        i += 1;
        let write = i == 100;

        let (jpeg, meta) = stream.next().unwrap();
        let start = Instant::now();

        println!(
            "Buffer size: {}, seq: {}, timestamp: {}",
            jpeg.len(),
            meta.sequence,
            meta.timestamp
        );
        use opencv::{core::*, imgproc::*};

        let mut decoder = JpegDecoder::new(jpeg);
        // decode the file
        let mut pixels = decoder.decode().unwrap();
        dbg!(pixels.len());

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

        opencv::imgproc::cvt_color(&rgb, &mut rgba, COLOR_RGB2RGBA, 0);

        if write {
            let _ = opencv::imgcodecs::imwrite(
                &format!("frame.bmp"),
                &rgba,
                &opencv::core::Vector::new(),
            );
        }

        let len = unsafe { rgba.dataend().offset_from(rgba.datastart()) } as usize;
        let rgba_data = unsafe { std::slice::from_raw_parts(rgba.datastart(), len) }.to_owned();
        let rgba8 = image::RgbaImage::from_raw(width, height, rgba_data).unwrap();

        let img = DynamicImage::ImageRgba8(rgba8);
        let ys = model.run(&[img])?;
        for ys in ys {
            for (i, mask) in ys.masks.iter().enumerate() {
                dbg!(mask.width(), mask.height());
                let mut pixels = Vec::new();
                for (x, y, p) in mask.enumerate_pixels() {
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

                opencv::imgproc::cvt_color(&greyscale, &mut mask_rgba, COLOR_GRAY2RGBA, 0);

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
                //opencv::core::mask(rgba, greyscale).unwrap();

                if write {
                    let _ = opencv::imgcodecs::imwrite(
                        &format!("mask_rgba_{i}.bmp"),
                        &mask_rgba,
                        &opencv::core::Vector::new(),
                    );

                    let _ = opencv::imgcodecs::imwrite(
                        &format!("mask_{i}.bmp"),
                        &greyscale,
                        &opencv::core::Vector::new(),
                    );
                    let _ = opencv::imgcodecs::imwrite(
                        &format!("masked_{i}.bmp"),
                        &masked,
                        &opencv::core::Vector::new(),
                    );
                }
            }
        }

        println!("Observed latency: {:?}", start.elapsed());

        if write {
            break;
        }
    }

    Ok(())
}
