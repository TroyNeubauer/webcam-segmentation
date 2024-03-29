#![allow(clippy::type_complexity)]

use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{s, Array, Axis, IxDyn};

use crate::{
    non_max_suppression, Args, Batch, Bbox, OrtBackend, OrtConfig, OrtEP, Point2, YOLOResult,
    YOLOTask,
};

pub struct YOLOv8 {
    // YOLOv8 model for all yolo-tasks
    engine: OrtBackend,
    nc: u32,
    nk: u32,
    nm: u32,
    height: u32,
    width: u32,
    batch: u32,
    task: YOLOTask,
    conf: f32,
    kconf: f32,
    iou: f32,
    names: Vec<String>,
    profile: bool,
}

impl YOLOv8 {
    pub fn new(config: Args) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::Cuda(config.device_id)
        } else {
            OrtEP::Cpu
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: config.task,
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let nk = 0;
        let nm = engine.nm();

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            kconf: config.kconf,
            iou: config.iou,
            profile: config.profile,
            nc,
            nk,
            nm,
            height,
            width,
            batch,
            task,
        })
    }

    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    pub fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Array<f32, IxDyn>> {
        let mut ys =
            Array::ones((xs.len(), 3, self.height() as usize, self.width() as usize)).into_dyn();
        ys.fill(144.0 / 255.0);
        for (idx, x) in xs.iter().enumerate() {
            let img = match self.task() {
                _ => {
                    let (w0, h0) = x.dimensions();
                    let w0 = w0 as f32;
                    let h0 = h0 as f32;
                    let (_, w_new, h_new) =
                        self.scale_wh(w0, h0, self.width() as f32, self.height() as f32); // f32 round
                    x.resize_exact(
                        w_new as u32,
                        h_new as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                }
            };

            for (x, y, rgb) in img.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        Ok(ys)
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<YOLOResult>> {
        // pre-process
        let t_pre = std::time::Instant::now();
        let xs_ = self.preprocess(xs)?;
        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        // run
        let t_run = std::time::Instant::now();
        let ys = self.engine.run(xs_, self.profile)?;
        if self.profile {
            println!("[Model Inference]: {:?}", t_run.elapsed());
        }

        // post-process
        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs)?;
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }

        Ok(ys)
    }

    pub fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<YOLOResult>> {
        const CXYWH_OFFSET: usize = 4; // cxcywh
        let preds = &xs[0];
        let protos = {
            if xs.len() > 1 {
                Some(&xs[1])
            } else {
                None
            }
        };
        let mut ys = Vec::new();
        for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
            // [bs, 4 + nc + nm, anchors]
            // input image
            let width_original = xs0[idx].width() as f32;
            let height_original = xs0[idx].height() as f32;
            let ratio =
                (self.width() as f32 / width_original).min(self.height() as f32 / height_original);

            // save each result
            let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
            for pred in anchor.axis_iter(Axis(1)) {
                // split preds for different tasks
                let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc() as usize]);
                let coefs = { Some(pred.slice(s![pred.len() - self.nm() as usize..]).to_vec()) };

                // confidence and id
                let (id, &confidence) = clss
                    .into_iter()
                    .enumerate()
                    .reduce(|max, x| if x.1 > max.1 { x } else { max })
                    .unwrap(); // definitely will not panic!

                // confidence filter
                if confidence < self.conf {
                    continue;
                }

                // bbox re-scale
                let cx = bbox[0] / ratio;
                let cy = bbox[1] / ratio;
                let w = bbox[2] / ratio;
                let h = bbox[3] / ratio;
                let x = cx - w / 2.;
                let y = cy - h / 2.;
                let y_bbox = Bbox::new(
                    x.max(0.0f32).min(width_original),
                    y.max(0.0f32).min(height_original),
                    w,
                    h,
                    id,
                    confidence,
                );

                // data merged
                data.push((y_bbox, None, coefs));
            }

            // nms
            non_max_suppression(&mut data, self.iou);

            // decode
            let mut y_bboxes: Vec<Bbox> = Vec::new();
            let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
            let mut masks = Vec::new();
            for elem in data.into_iter() {
                if let Some(kpts) = elem.1 {
                    y_kpts.push(kpts)
                }

                // decode masks
                if let Some(coefs) = elem.2 {
                    let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                    let (nm, nh, nw) = proto.dim();

                    // coefs * proto -> mask
                    let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                    let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                    let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)
                                                                           // build image from ndarray
                    let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                        match ImageBuffer::from_raw(nw as u32, nh as u32, mask.into_raw_vec()) {
                            Some(image) => image,
                            None => panic!("can not create image from ndarray"),
                        };
                    let mut mask_im = image::DynamicImage::from(mask_im); // -> dyn

                    // rescale masks
                    let (_, w_mask, h_mask) =
                        self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                    let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                    let mask_original = mask_cropped.resize_exact(
                        // resize_to_fill
                        width_original as u32,
                        height_original as u32,
                        match self.task() {
                            YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                        },
                    );

                    // crop-mask with bbox
                    let mut mask_original_cropped = mask_original.into_luma8();
                    for y in 0..height_original as usize {
                        for x in 0..width_original as usize {
                            let padding = 10.0;
                            let xmin = elem.0.xmin - padding;
                            let xmax = elem.0.xmax() + padding;
                            let ymin = elem.0.ymin - padding;
                            let ymax = elem.0.ymax() + padding;
                            if x < xmin as usize
                                || x > xmax as usize
                                || y < ymin as usize
                                || y > ymax as usize
                            {
                                mask_original_cropped.put_pixel(
                                    x as u32,
                                    y as u32,
                                    image::Luma([0u8]),
                                );
                            }
                        }
                    }
                    masks.push(mask_original_cropped);
                }
                y_bboxes.push(elem.0);
            }

            // save each result
            let y = YOLOResult {
                probs: None,
                bboxes: y_bboxes,
                keypoints: y_kpts,
                masks,
            };
            ys.push(y);
        }

        Ok(ys)
    }

    pub fn summary(&self) {
        println!(
            "\nSummary:\n\
            > Task: {:?}{}\n\
            > EP: {:?} {}\n\
            > Dtype: {:?}\n\
            > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
            > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
            ",
            self.task(),
            match self.engine.author().zip(self.engine.version()) {
                Some((author, ver)) => format!(" ({} {})", author, ver),
                None => String::from(""),
            },
            self.engine.ep(),
            if let OrtEP::Cpu = self.engine.ep() {
                ""
            } else {
                "(May still fall back to CPU)"
            },
            self.engine.dtype(),
            self.batch(),
            if self.engine.is_batch_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.height(),
            if self.engine.is_height_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.width(),
            if self.engine.is_width_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.nc(),
            self.nk(),
            self.nm(),
            self.conf,
            self.kconf,
            self.iou,
        );
    }

    pub fn engine(&self) -> &OrtBackend {
        &self.engine
    }

    pub fn conf(&self) -> f32 {
        self.conf
    }

    pub fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    pub fn conf_mut(&mut self) -> &mut f32 {
        &mut self.conf
    }

    pub fn kconf(&self) -> f32 {
        self.kconf
    }

    pub fn iou(&self) -> f32 {
        self.iou
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn nc(&self) -> u32 {
        self.nc
    }

    pub fn nk(&self) -> u32 {
        self.nk
    }

    pub fn nm(&self) -> u32 {
        self.nm
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
    }
}
