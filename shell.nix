{ pkgs ? (import <nixpkgs> { 
    config.allowUnfree = true;
}), ... }:

let
  lib-path = with pkgs; lib.makeLibraryPath [
    openssl
    cudatoolkit
  ];

in pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    gcc12
    pkg-config

    onnxruntime
    openssl
    linuxPackages.nvidia_x11
    cudatoolkit
    cudaPackages.cuda_nvrtc
    libGLU libGL
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${lib-path}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_nvrtc}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcurand}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.tensorrt_8_6}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.gcc-unwrapped.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.onnxruntime}/lib:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH
    export CUDA_PATH=${pkgs.cudatoolkit}
  '';
}
