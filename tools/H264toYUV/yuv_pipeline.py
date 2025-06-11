#!/usr/bin/env python3
import os
import argparse
import subprocess

def run_pristine_decode(input_dir, resolutions, output_dir):
    for res in resolutions:
        width, height = res.split('x')
        out_path = os.path.join(output_dir, res)
        os.makedirs(out_path, exist_ok=True)
        for f in os.listdir(input_dir):
            if f.endswith(".264"):
                input_path = os.path.join(input_dir, f)
                yuv_name = os.path.splitext(f)[0] + ".yuv"
                output_path = os.path.join(out_path, yuv_name)
                cmd = f"ffmpeg -y -i {input_path} -vf scale={width}:{height}:flags=lanczos -pix_fmt yuv420p {output_path}"
                subprocess.run(cmd, shell=True)

def run_encode_yuv(input_root, crfs, output_root):
    for res in os.listdir(input_root):
        res_path = os.path.join(input_root, res)
        for f in os.listdir(res_path):
            if f.endswith(".yuv"):
                input_file = os.path.join(res_path, f)
                for crf in crfs:
                    crf_path = os.path.join(output_root, res, str(crf))
                    os.makedirs(crf_path, exist_ok=True)
                    out264 = os.path.join(crf_path, f.replace(".yuv", ".264"))
                    width, height = res.split('x')
                    cmd = f"ffmpeg -y -s {width}x{height} -pix_fmt yuv420p -i {input_file} -c:v libx264 -preset slow -crf {crf} -an -f h264 {out264}"
                    subprocess.run(cmd, shell=True)

def run_decode_encoded(input_root, output_root):
    for res in os.listdir(input_root):
        res_path = os.path.join(input_root, res)
        for crf in os.listdir(res_path):
            crf_path = os.path.join(res_path, crf)
            for f in os.listdir(crf_path):
                if f.endswith(".264"):
                    input_file = os.path.join(crf_path, f)
                    out_yuv_path = os.path.join(output_root, res, crf)
                    os.makedirs(out_yuv_path, exist_ok=True)
                    out_yuv = os.path.join(out_yuv_path, f.replace(".264", ".yuv"))
                    width, height = res.split('x')
                    cmd = f"ffmpeg -y -i {input_file} -pix_fmt yuv420p -s {width}x{height} {out_yuv}"
                    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YUV Processing Pipeline for BVQA Dataset Preparation")
    parser.add_argument('--pristine_dir', type=str, required=True, help='Input directory for pristine .264 files')
    parser.add_argument('--resolutions', nargs='+', required=True, help='Resolutions to scale to, e.g., 1920x1080 960x540')
    parser.add_argument('--pristine_yuv_out', type=str, required=True, help='Output directory for pristine .yuv files')
    parser.add_argument('--crf_levels', nargs='+', type=int, default=[20, 23, 25, 28, 30, 35, 40, 45], help='List of CRF levels for encoding')
    parser.add_argument('--encoded_out', type=str, required=True, help='Output directory for encoded .264 files')
    parser.add_argument('--decoded_out', type=str, required=True, help='Output directory for decoded .yuv files from encoded videos')

    args = parser.parse_args()

    print("Step 1: Decode pristine .264 to YUV420p")
    run_pristine_decode(args.pristine_dir, args.resolutions, args.pristine_yuv_out)

    print("Step 2: Encode pristine YUV to .264 with CRF levels")
    run_encode_yuv(args.pristine_yuv_out, args.crf_levels, args.encoded_out)

    print("Step 3: Decode encoded .264 back to YUV420p")
    run_decode_encoded(args.encoded_out, args.decoded_out)

    print("All steps completed.")
