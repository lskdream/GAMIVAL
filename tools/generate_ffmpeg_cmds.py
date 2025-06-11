# generate_ffmpeg_commands.py
# CLI tool to generate batch FFmpeg commands for scaling .264 videos into raw YUV format

import argparse
import os

def generate_commands(games, resolutions, output_dir, input_dir, suffix=".264"):
    commands = []
    for game in games:
        for width, height in resolutions:
            cmd = (
                f"ffmpeg -hide_banner -y -vsync 0 -i {input_dir}/{game}{suffix} "
                f"-filter:v 'scale={width}:{height}:flags=lanczos' -f rawvideo -pix_fmt yuv420p "
                f"{output_dir}/{game}_{width}x{height}.yuv420p < /dev/null"
            )
            commands.append(cmd)
    return commands

def main():
    parser = argparse.ArgumentParser(description="Generate FFmpeg commands for batch video scaling.")
    parser.add_argument('--games', nargs='+', required=True, help='List of video names (without extension)')
    parser.add_argument('--res', nargs='+', required=True, help='Resolutions in WxH format, e.g. 720x1280')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .264 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output YUV files')
    parser.add_argument('--output_sh', type=str, default='run_ffmpeg.sh', help='Shell script file to save commands')
    args = parser.parse_args()

    # Parse resolution list
    resolutions = [(int(r.split('x')[0]), int(r.split('x')[1])) for r in args.res]

    # Generate commands
    commands = generate_commands(args.games, resolutions, args.output_dir, args.input_dir)

    # Write to .sh file
    with open(args.output_sh, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")

    os.chmod(args.output_sh, 0o755)
    print(f"✅ Shell script saved to {args.output_sh} with {len(commands)} commands.")

if __name__ == '__main__':
    main()
