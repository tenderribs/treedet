import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver.

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, demo_postprocess

TRT_LOGGER = trt.Logger()

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def main(args):
    trt_runtime = trt.Runtime(TRT_LOGGER)

    engine = load_engine(trt_runtime, args.model)
    context = engine.create_execution_context()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    image_files = [f for f in os.listdir(args.images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(args.images_path, f))]

    total_inference = 0

    for image_file in image_files:
        origin_img = cv2.imread(os.path.join(args.images_path, image_file))
        img, ratio = preprocess(origin_img, input_shape)

        # Allocate device memory
        d_input = cuda.mem_alloc(int(1 * img.nbytes))
        d_output = cuda.mem_alloc(int(1 * np.prod(engine.get_binding_shape(1)) * 4))  # Assuming output is float32
        bindings = [int(d_input), int(d_output)]

        # Transfer input data to device
        cuda.memcpy_htod(d_input, img.ravel())

        # Execute model
        start = time.perf_counter()
        context.execute_v2(bindings=bindings)
        total_inference += (time.perf_counter() - start)

        # Fetch output from device
        output = np.empty(engine.get_binding_shape(1), dtype=np.float32)  # Assuming output is float32
        cuda.memcpy_dtoh(output, d_output)

        # Process the output as per your requirements...
        # Note: You'll need to adjust the post-processing code below to fit your model's specific output format

        # mkdir(args.output_dir)
        # output_path = os.path.join(args.output_dir, image_file)
        # cv2.imwrite(output_path, origin_img)

    print(f"Total inference time: {total_inference}s for {len(image_files)} images, avg {total_inference / len(image_files)}s per image")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TensorRT inference")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the TensorRT engine file.")
    parser.add_argument("--images_path", type=str, required=True, help="Path to your input images directory.")
    parser.add_argument("-o", "--output_dir", type=str, default='demo_output', help="Path to your output directory.")
    parser.add_argument("--input_shape", type=str, default="384,672", help="Specify an input shape for inference.")
    parser.add_argument("--score_thr", type=float, default=0.8, help="Score threshold to filter the results.")

    args = parser.parse_args()
    main(args)
