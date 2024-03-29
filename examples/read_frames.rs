use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

fn main() {
    opencv::core::set_num_threads(1).unwrap();
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

    // The actual format chosen by the device driver may differ from what we
    // requested! Print it out to get an idea of what is actually used now.
    println!("Format in use:\n{}", fmt);

    // Now we'd like to capture some frames!
    // First, we need to create a stream to read buffers from. We choose a
    // mapped buffer stream, which uses mmap to directly access the device
    // frame buffer. No buffers are copied nor allocated, so this is actually
    // a zero-copy operation.

    // To achieve the best possible performance, you may want to use a
    // UserBufferStream instance, but this is not supported on all devices,
    // so we stick to the mapped case for this example.
    // Please refer to the rustdoc docs for a more detailed explanation about
    // buffer transfers.

    // Create the stream, which will internally 'allocate' (as in map) the
    // number of requested buffers for us.
    let mut stream = Stream::with_buffers(&mut dev, Type::VideoCapture, 4)
        .expect("Failed to create buffer stream");

    // At this point, the stream is ready and all buffers are setup.
    // We can now read frames (represented as buffers) by iterating through
    // the stream. Once an error condition occurs, the iterator will return
    // None.
    loop {
        let (jpeg, meta) = stream.next().unwrap();
        println!(
            "Buffer size: {}, seq: {}, timestamp: {}",
            jpeg.len(),
            meta.sequence,
            meta.timestamp
        );
        use opencv::{core::*, imgproc::*};

        let mut pixels = turbojpeg::decompress_image::<image::Rgb<u8>>(jpeg)
            .unwrap()
            .into_vec();
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

        opencv::imgproc::cvt_color(&rgb, &mut rgba, COLOR_RGB2RGBA, 0).unwrap();

        opencv::imgcodecs::imwrite("out.bmp", &rgba, &opencv::core::Vector::new()).unwrap();
        break;

        // To process the captured data, you can pass it somewhere else.
        // If you want to modify the data or extend its lifetime, you have to
        // copy it. This is a best-effort tradeoff solution that allows for
        // zero-copy readers while enforcing a full clone of the data for
        // writers.
    }
}
