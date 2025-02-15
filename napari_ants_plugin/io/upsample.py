#!/usr/bin/env python3
"""
Atlas Upsampling Script

This module implements a cube‐wise atlas annotation upsampling routine using batched PyTorch
interpolation, asynchronous disk writes (via Zarr), and a threaded CPU buffer.

Command-line options allow you to override the default parameters including the mapping
spaces and whether to swap the X and Y axes.
"""

import logging
import time
import threading
import queue
import argparse
from typing import List, Tuple, Optional

import numpy as np
import zarr
import ants
import torch
import torch.nn.functional as F
from tqdm import tqdm
from zarr.storage import LocalStore
import brainglobe_space as bg


def setup_logger(name: str = "AtlasUpsampler", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class AtlasUpsampler:
    """
    Upsamples an atlas annotation in cubes using PyTorch for interpolation and writes
    the final upsampled volume into a Zarr dataset.

    Parameters:
        annotation_file: Path to the input atlas annotation (e.g. NIfTI file).
        final_zarr_path: Output path for the final Zarr dataset.
        atlas_voxel_size: Voxel size for the atlas (in Z, Y, X order).
        raw_voxel_size: Voxel size for the raw data (in Z, Y, X order).
        cube_divisor: Factor to divide the image dimensions to define the base cube size.
        batch_size: Number of cubes to process in each PyTorch interpolation batch.
        src_space: Optional source space for mapping (e.g., 'asr').
        dest_space: Optional destination space for mapping (e.g., 'iar').
        swap_xy: If True, swap the X and Y axes.
    """

    def __init__(
        self,
        annotation_file: str = 'transformed_annotation.nii.gz',
        final_zarr_path: str = 'resampled_full_anno.zarr',
        atlas_voxel_size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
        raw_voxel_size: Tuple[float, float, float] = (4.0, 2.0, 2.0),
        cube_divisor: int = 10,
        batch_size: int = 16,
        src_space: Optional[str] = None,
        dest_space: Optional[str] = None,
        swap_xy: bool = False,  # By default, do not swap axes.
    ):
        self.annotation_file = annotation_file
        self.final_zarr_path = final_zarr_path
        self.atlas_voxel_size = np.array(atlas_voxel_size)
        self.raw_voxel_size = np.array(raw_voxel_size)
        self.cube_divisor = cube_divisor
        self.batch_size = batch_size
        self.src_space = src_space
        self.dest_space = dest_space
        self.swap_xy = swap_xy
        self.logger = setup_logger("AtlasUpsampler")
        self.final_array: Optional[zarr.core.Array] = None
        self.write_queue: Optional[queue.Queue] = None
        self.final_array_lock: Optional[threading.Lock] = None

    def process_batch_torch(
        self,
        cube_list: List[np.ndarray],
        target_shapes: List[Tuple[int, int, int]],
        full_target_shape: Tuple[int, int, int]
    ) -> List[np.ndarray]:
        """
        Process a batch of cubes using PyTorch interpolation.

        Each cube is assumed to be padded to the same base shape.

        Args:
            cube_list: List of padded cubes (each a numpy array).
            target_shapes: List of desired (cropped) target shapes.
            full_target_shape: Target shape after interpolation for a full (padded) cube.

        Returns:
            List of upsampled cubes, cropped to their desired target shape.
        """
        cubes_tensor = torch.stack(
            [torch.tensor(cube.copy(), device='cuda') for cube in cube_list]
        ).unsqueeze(1).float()

        cubes_tensor = cubes_tensor.to(memory_format=torch.channels_last_3d)

        start = time.time()
        with torch.amp.autocast('cuda'):
            up_tensor = F.interpolate(
                cubes_tensor, size=full_target_shape, mode='nearest')
        torch.cuda.synchronize()  # Wait for GPU kernels to finish.
        interp_time = time.time() - start
        self.logger.info(
            f"Batch interpolation time: {interp_time:.3f} seconds for batch size {cubes_tensor.shape[0]} "
            f"with full target shape {full_target_shape}."
        )

        up_cubes = up_tensor.squeeze(1).cpu().numpy()

        cropped_results = []
        for up_cube, tgt_shape in zip(up_cubes, target_shapes):
            slices = tuple(slice(0, t) for t in tgt_shape)
            cropped_results.append(up_cube[slices])
        return cropped_results

    def _writer_thread(self) -> None:
        """
        Thread worker function to write upsampled cubes to the final Zarr array.
        """
        assert self.write_queue is not None and self.final_array is not None
        while True:
            item = self.write_queue.get()
            if item is None:  # Sentinel to signal termination.
                self.write_queue.task_done()
                break
            up_cube, coords = item
            with self.final_array_lock:
                out_z0, out_z1, out_y0, out_y1, out_x0, out_x1 = coords
                self.final_array[out_z0:out_z1,
                                 out_y0:out_y1, out_x0:out_x1] = up_cube
            self.write_queue.task_done()

    def run(self) -> None:
        """
        Execute the full atlas upsampling pipeline.
        """
        self.logger.info("=== Starting Cube-wise Upsampling ===")
        self.logger.info(
            f"Loading atlas annotation from '{self.annotation_file}'...")
        atlas_img = ants.image_read(self.annotation_file)
        atlas_data = atlas_img.numpy()

        # Map atlas data if both source and destination spaces are provided.
        if self.src_space and self.dest_space:
            self.logger.info(
                f"Mapping atlas data from {self.src_space} to {self.dest_space} space...")
            atlas_data = bg.map_stack_to(
                self.src_space, self.dest_space, atlas_data)
        else:
            self.logger.info(
                "No mapping performed; src_space and dest_space not provided.")

        # Optionally swap the X and Y axes.
        if self.swap_xy:
            self.logger.info("Swapping X and Y axes...")
            atlas_data = atlas_data.transpose(0, 2, 1)
        else:
            self.logger.info("Not swapping X and Y axes.")

        # Flip the left-right axis.
        atlas_data = np.flip(atlas_data, axis=2)
        in_z, in_y, in_x = atlas_data.shape
        self.logger.info(
            f"Atlas annotation shape (Z, Y, X) after processing: ({in_z}, {in_y}, {in_x})")

        new_zoom_factors = (
            self.atlas_voxel_size[0] / self.raw_voxel_size[0],  # Z
            # New Y from original X
            self.atlas_voxel_size[2] / self.raw_voxel_size[2],
            # New X from original Y
            self.atlas_voxel_size[1] / self.raw_voxel_size[1]
        )
        self.logger.info(f"New zoom factors (Z, Y, X): {new_zoom_factors}")

        cube_size = (
            max(1, in_z // self.cube_divisor),
            max(1, in_y // self.cube_divisor),
            max(1, in_x // self.cube_divisor)
        )
        self.logger.info(
            f"Processing atlas in cubes of base size (Z, Y, X): {cube_size}")

        final_shape = (
            int(round(in_z * new_zoom_factors[0])),
            int(round(in_y * new_zoom_factors[1])),
            int(round(in_x * new_zoom_factors[2]))
        )
        self.logger.info(f"Final upsampled shape (Z, Y, X): {final_shape}")

        chunks = (
            min(64, final_shape[0]),
            min(256, final_shape[1]),
            min(256, final_shape[2])
        )
        store_final = LocalStore(self.final_zarr_path)
        root_final = zarr.group(store=store_final, overwrite=True)
        self.final_array = root_final.create_array(
            'data', shape=final_shape, chunks=chunks, dtype=np.uint16)
        self.logger.info(
            f"Created final Zarr dataset at '{self.final_zarr_path}' with chunk shape {chunks}.")

        full_target_shape = (
            int(round(cube_size[0] * new_zoom_factors[0])),
            int(round(cube_size[1] * new_zoom_factors[1])),
            int(round(cube_size[2] * new_zoom_factors[2]))
        )

        self.write_queue = queue.Queue()
        self.final_array_lock = threading.Lock()

        writer = threading.Thread(target=self._writer_thread, daemon=True)
        writer.start()

        batch_cubes: List[np.ndarray] = []
        batch_desired_target_shapes: List[Tuple[int, int, int]] = []
        batch_coords: List[Tuple[int, int, int, int, int, int]] = []
        cube_counter = 0
        total_cubes = (
            ((in_z - 1) // cube_size[0] + 1) *
            ((in_y - 1) // cube_size[1] + 1) *
            ((in_x - 1) // cube_size[2] + 1)
        )

        def flush_batch() -> None:
            nonlocal batch_cubes, batch_desired_target_shapes, batch_coords, cube_counter
            if batch_cubes:
                self.logger.info(
                    f"Batch processing {len(batch_cubes)} cubes with padded input shape {cube_size} -> "
                    f"full target shape {full_target_shape}"
                )
                up_cubes_list = self.process_batch_torch(
                    batch_cubes, batch_desired_target_shapes, full_target_shape)
                for up_cube, coords in zip(up_cubes_list, batch_coords):
                    self.write_queue.put((up_cube, coords))
                cube_counter += len(up_cubes_list)
                self.logger.info(
                    f"Processed {cube_counter} / {total_cubes} cubes (queued for disk write).")
                batch_cubes.clear()
                batch_desired_target_shapes.clear()
                batch_coords.clear()

        for z0 in tqdm(range(0, in_z, cube_size[0]), desc="Z-slices"):
            for y0 in range(0, in_y, cube_size[1]):
                for x0 in range(0, in_x, cube_size[2]):
                    z1 = min(z0 + cube_size[0], in_z)
                    y1 = min(y0 + cube_size[1], in_y)
                    x1 = min(x0 + cube_size[2], in_x)
                    out_z0 = int(round(z0 * new_zoom_factors[0]))
                    out_z1 = int(round(z1 * new_zoom_factors[0]))
                    out_y0 = int(round(y0 * new_zoom_factors[1]))
                    out_y1 = int(round(y1 * new_zoom_factors[1]))
                    out_x0 = int(round(x0 * new_zoom_factors[2]))
                    out_x1 = int(round(x1 * new_zoom_factors[2]))
                    cube = atlas_data[z0:z1, y0:y1, x0:x1]
                    desired_target_shape = (
                        out_z1 - out_z0, out_y1 - out_y0, out_x1 - out_x0)
                    pad_z = cube_size[0] - cube.shape[0]
                    pad_y = cube_size[1] - cube.shape[1]
                    pad_x = cube_size[2] - cube.shape[2]
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        cube = np.pad(cube,
                                      pad_width=(
                                          (0, pad_z), (0, pad_y), (0, pad_x)),
                                      mode='edge')
                    batch_cubes.append(cube)
                    batch_desired_target_shapes.append(desired_target_shape)
                    batch_coords.append(
                        (out_z0, out_z1, out_y0, out_y1, out_x0, out_x1))
                    if len(batch_cubes) >= self.batch_size:
                        flush_batch()
            if batch_cubes:
                flush_batch()
            self.logger.info(
                f"Finished Z-slice {z0} to {min(z0 + cube_size[0], in_z)}. Waiting for disk writes..."
            )
            self.write_queue.join()

        self.write_queue.put(None)
        writer.join()
        self.logger.info(
            "Upsampling complete. Final upsampled atlas annotation saved in Zarr format.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cube-wise Atlas Upsampling using PyTorch, Zarr, and asynchronous disk writes.'
    )
    parser.add_argument(
        '--annotation-file',
        type=str,
        default='transformed_annotation.nii.gz',
        help='Input atlas annotation file (e.g., NIfTI file).'
    )
    parser.add_argument(
        '--final-zarr-path',
        type=str,
        default='resampled_full_anno.zarr',
        help='Output path for the final Zarr dataset.'
    )
    parser.add_argument(
        '--atlas-voxel-size',
        type=str,
        default='25.0,25.0,25.0',
        help='Atlas voxel size in Z,Y,X order (comma-separated, e.g., "25.0,25.0,25.0").'
    )
    parser.add_argument(
        '--raw-voxel-size',
        type=str,
        default='4.0,2.0,2.0',
        help='Raw voxel size in Z,Y,X order (comma-separated, e.g., "4.0,2.0,2.0").'
    )
    parser.add_argument(
        '--cube-divisor',
        type=int,
        default=10,
        help='Cube divisor: base cube size is computed as (dimension // cube_divisor).'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of cubes to process in each batch.'
    )
    parser.add_argument(
        '--src-space',
        type=str,
        default='asr',
        help='Optional source space for mapping (e.g., "asr"). If not provided, mapping is skipped.'
    )
    parser.add_argument(
        '--dest-space',
        type=str,
        default="iar",
        help='Optional destination space for mapping (e.g., "iar"). If not provided, mapping is skipped.'
    )
    # By default, do not swap the X and Y axes. Use --swap-xy to enable swapping.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--swap-xy',
        dest='swap_xy',
        action='store_true',
        help='Swap the X and Y axes.'
    )
    group.add_argument(
        '--no-swap-xy',
        dest='swap_xy',
        action='store_false',
        help='Do not swap the X and Y axes (default).'
    )
    parser.set_defaults(swap_xy=False)

    args = parser.parse_args()

    atlas_voxel_size = tuple(map(float, args.atlas_voxel_size.split(',')))
    raw_voxel_size = tuple(map(float, args.raw_voxel_size.split(',')))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your GPU settings.")

    upsampler = AtlasUpsampler(
        annotation_file=args.annotation_file,
        final_zarr_path=args.final_zarr_path,
        atlas_voxel_size=atlas_voxel_size,
        raw_voxel_size=raw_voxel_size,
        cube_divisor=args.cube_divisor,
        batch_size=args.batch_size,
        src_space=args.src_space,
        dest_space=args.dest_space,
        swap_xy=args.swap_xy
    )
    upsampler.run()


if __name__ == "__main__":
    main()
