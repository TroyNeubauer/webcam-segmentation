{
  pkgs,
  src,
  rustPlatform,
}:
rustPlatform.buildRustPackage rec {
  inherit src;

  pname = "webcam-segmentation";
  version = "0.1.0";

  cargoLock = {
    lockFile = ./Cargo.lock;
  };

  buildInputs = with pkgs; [
   # libjpeg
   libjpeg

   v4l-utils
   # candle / opencv
   onnxruntime
   openssl
   linuxPackages.nvidia_x11
   cudatoolkit
   cudaPackages.cuda_nvrtc
   cudaPackages.cudnn
   cudaPackages.libcurand
   cudaPackages.tensorrt
   cudaPackages.libcublas
   clang.cc.lib
   opencv
   libGLU
   libGL
 ];

 nativeBuildInputs = with pkgs; [
   autoPatchelfHook
   cmake
   clang
   gcc12
   nasm
   pkg-config
 ];

  preBuild = ''
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_nvrtc}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcurand}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.tensorrt}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.gcc-unwrapped.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.onnxruntime}/lib:$LD_LIBRARY_PATH
    export CUDA_PATH=${pkgs.cudatoolkit}
    # opencv-rs
    
    export PATH=${pkgs.llvmPackages.clang}/bin:$PATH
    export LIBCLANG_PATH=${pkgs.llvmPackages.clang.cc.lib}/lib
    export LD_LIBRARY_PATH=${pkgs.llvmPackages.clang.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.opencv}/lib:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=${pkgs.opencv}:PKG_CONFIG_PATH
  '';

  meta = {
    description = "A Rust based service that crops the background from a live video device using v4l and YOLOv8";
    homepage = "https://github.com/TroyNeubauer/webcam-segmentation";
  };
}
