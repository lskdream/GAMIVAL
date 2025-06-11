import argparse
import numpy as np
import cv2
import os

def read_yuv420_frame(filename, width, height, frame_idx=0):
    frame_size = width * height * 3 // 2
    with open(filename, 'rb') as f:
        f.seek(frame_idx * frame_size)
        yuv = np.frombuffer(f.read(frame_size), dtype=np.uint8)
        if yuv.size < frame_size:
            raise ValueError("Incomplete frame or frame index out of bounds.")
        y = yuv[0:width*height].reshape((height, width))
        u = yuv[width*height:width*height + width*height//4].reshape((height//2, width//2))
        v = yuv[width*height + width*height//4:].reshape((height//2, width//2))
        u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        yuv_img = cv2.merge((y, u_up, v_up))
        bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        return bgr_img

def main():
    parser = argparse.ArgumentParser(description='Display a frame from a YUV 4:2:0 file.')
    parser.add_argument('--input', required=True, help='Path to YUV file')
    parser.add_argument('--width', type=int, required=True, help='Width of the video')
    parser.add_argument('--height', type=int, required=True, help='Height of the video')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to display')
    args = parser.parse_args()

    img = read_yuv420_frame(args.input, args.width, args.height, args.frame)
    cv2.imshow(f'Frame {args.frame}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
