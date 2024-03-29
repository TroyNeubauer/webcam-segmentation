{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system;
          config = {
            cudaSupport = true;
            allowBroken = true;
            allowUnfree = true;
          };
        };
        rust = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            gcc12
            pkg-config

            # libjpeg
            cmake
            nasm
            libjpeg

            v4l-utils
            onnxruntime
            openssl
            linuxPackages.nvidia_x11
            cudatoolkit
            cudaPackages.cuda_nvrtc
            cudaPackages.cudnn
            cudaPackages.libcurand
            cudaPackages.tensorrt
            cudaPackages.libcublas
            clang
            clang.cc.lib
            opencv
            libGLU
            libGL
          ];

          shellHook = ''
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
        };
      }
    );
}
