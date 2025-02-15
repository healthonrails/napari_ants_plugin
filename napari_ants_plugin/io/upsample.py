#!/usr/bin/env python3
import os
import logging
import time
import threading
import queue
import numpy as np
import zarr
import ants
import torch
import torch.nn.functional as F
from tqdm import tqdm
from zarr.storage import LocalStore
import brainglobe_space as bg


def setup_logger() -> logging.Logger:
    """Set up and return a console logger."""
    logger = logging.getLogger("AtlasUpsampler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def process_batch_torch(cube_list, target_shapes, full_target_shape):
    """
    Process a batch of cubes (all padded to the same full input shape) using PyTorch interpolation.

    Parameters:
        cube_list (list[np.ndarray]): List of padded cubes, each with shape equal to the full cube size.
        target_shapes (list[tuple]): For each cube, the desired output shape (may be smaller than full_target_shape).
        full_target_shape (tuple): The upsampled shape corresponding to a full (padded) cube.

    Returns:
        list[np.ndarray]: List of upsampled cubes, each cropped to its desired target shape.
    """
    # Convert each cube to a torch tensor (copy to ensure positive strides)
    cubes_tensor = torch.stack(
        [torch.tensor(cube.copy(), device='cuda') for cube in cube_list]
    )
    # Add channel dimension: (N, 1, D, H, W)
    cubes_tensor = cubes_tensor.unsqueeze(1).float()
    # Convert to channels_last_3d format for optimal GPU performance.
    cubes_tensor = cubes_tensor.to(memory_format=torch.channels_last_3d)

    # Benchmark the interpolation call.
    start = time.time()
    with torch.amp.autocast('cuda'):
        up_tensor = F.interpolate(
            cubes_tensor, size=full_target_shape, mode='nearest')
    torch.cuda.synchronize()  # Ensure GPU kernels are done.
    interp_time = time.time() - start
    logging.getLogger("AtlasUpsampler").info(
        f"Batch interpolation time: {interp_time:.3f} seconds for batch size {cubes_tensor.shape[0]} with full target shape {full_target_shape}."
    )

    # Remove channel dimension; shape: (N, full_target_D, full_target_H, full_target_W)
    up_cubes = up_tensor.squeeze(1).cpu().numpy()

    # Crop each upsampled cube to its desired target shape.
    cropped_results = []
    for up_cube, tgt_shape in zip(up_cubes, target_shapes):
        slices = tuple(slice(0, t) for t in tgt_shape)
        cropped_results.append(up_cube[slices])
    return cropped_results


def main():
    logger = setup_logger()
    logger.info(
        "=== Starting Cube-wise Upsampling (Batched PyTorch with CPU Buffer and Z-slice Flushing) ===")

    # File paths.
    annotation_file = 'transformed_annotation.nii.gz'
    final_zarr_path = 'resampled_full_anno.zarr'

    # Define voxel sizes (in (Z, Y, X) order).
    atlas_voxel_size = np.array([25.0, 25.0, 25.0])
    raw_voxel_size = np.array([4.0,  2.0,  2.0])

    # Load atlas annotation using ANTs.
    logger.info(f"Loading atlas annotation from '{annotation_file}'...")
    atlas_img = ants.image_read(annotation_file)
    atlas_data = atlas_img.numpy()

    # Map atlas data from ASR to IAL space.
    logger.info("Mapping atlas data from ASR to IAL space...")
    atlas_data = bg.map_stack_to('asr', 'iar', atlas_data)
    # --- Swap the X and Y axes ---
    # Original shape is (Z, Y, X); swap Y and X so new shape becomes (Z, new_Y, new_X)
    atlas_data = atlas_data.transpose(0, 2, 1)
    # flip the left right axis
    atlas_data = np.flip(atlas_data, axis=2)
    in_z, in_y, in_x = atlas_data.shape
    logger.info(
        f"Swapped atlas annotation shape (Z, Y, X): ({in_z}, {in_y}, {in_x})")

    # Compute new zoom factors (after swapping):
    # new_zoom_factors = (atlas_voxel_Z/raw_voxel_Z,
    #                     atlas_voxel_X/raw_voxel_X,  <-- new Y from original X
    #                     atlas_voxel_Y/raw_voxel_Y)  <-- new X from original Y
    new_zoom_factors = (
        atlas_voxel_size[0] / raw_voxel_size[0],  # Z
        atlas_voxel_size[2] / raw_voxel_size[2],  # New Y
        atlas_voxel_size[1] / raw_voxel_size[1]   # New X
    )
    logger.info(f"New zoom factors (Z, Y, X): {new_zoom_factors}")

    # Define a fixed cube (subvolume) size for processing.
    cube_size = (
        max(1, in_z // 10),
        max(1, in_y // 10),
        max(1, in_x // 10)
    )
    logger.info(
        f"Processing atlas in cubes of base size (Z, Y, X): {cube_size}")

    # Compute the final upsampled volume dimensions.
    out_z = int(round(in_z * new_zoom_factors[0]))
    out_y = int(round(in_y * new_zoom_factors[1]))
    out_x = int(round(in_x * new_zoom_factors[2]))
    final_shape = (out_z, out_y, out_x)
    logger.info(f"Final upsampled shape (Z, Y, X): {final_shape}")

    # Create final Zarr dataset.
    chunks = (
        min(64, final_shape[0]),
        min(256, final_shape[1]),
        min(256, final_shape[2])
    )
    store_final = LocalStore(final_zarr_path)
    root_final = zarr.group(store=store_final, overwrite=True)
    final_array = root_final.create_array(
        'data', shape=final_shape, chunks=chunks, dtype=np.uint16
    )
    logger.info(
        f"Created final Zarr dataset at '{final_zarr_path}' with chunk shape {chunks}.")

    # Compute the full target shape corresponding to a full (padded) cube.
    full_target_shape = (
        int(round(cube_size[0] * new_zoom_factors[0])),
        int(round(cube_size[1] * new_zoom_factors[1])),
        int(round(cube_size[2] * new_zoom_factors[2]))
    )

    # --- Set up a CPU memory buffer using a thread-safe queue for asynchronous disk writes ---
    write_queue = queue.Queue()
    final_array_lock = threading.Lock()

    def writer_thread():
        """Continuously write items from the queue to the final Zarr array."""
        while True:
            item = write_queue.get()
            if item is None:  # Sentinel to signal completion.
                write_queue.task_done()
                break
            up_cube, coords = item
            with final_array_lock:
                out_z0, out_z1, out_y0, out_y1, out_x0, out_x1 = coords
                final_array[out_z0:out_z1, out_y0:out_y1,
                            out_x0:out_x1] = up_cube
            write_queue.task_done()

    writer = threading.Thread(target=writer_thread)
    writer.start()

    # Set up batch parameters.
    BATCH_SIZE = 16
    batch_cubes = []                 # Padded cubes for current batch.
    batch_desired_target_shapes = []  # Their desired (cropped) target shapes.
    batch_coords = []                # Their final array coordinates.
    cube_counter = 0
    total_cubes = (
        ((in_z - 1) // cube_size[0] + 1) *
        ((in_y - 1) // cube_size[1] + 1) *
        ((in_x - 1) // cube_size[2] + 1)
    )

    def flush_batch():
        nonlocal batch_cubes, batch_desired_target_shapes, batch_coords, cube_counter
        if batch_cubes:
            logger.info(
                f"Batch processing {len(batch_cubes)} cubes with padded input shape {cube_size} -> full target shape {full_target_shape}")
            up_cubes_list = process_batch_torch(
                batch_cubes, batch_desired_target_shapes, full_target_shape)
            for up_cube, coords in zip(up_cubes_list, batch_coords):
                write_queue.put((up_cube, coords))
            cube_counter += len(up_cubes_list)
            logger.info(
                f"Processed {cube_counter} / {total_cubes} cubes (queued for disk write).")
            batch_cubes.clear()
            batch_desired_target_shapes.clear()
            batch_coords.clear()

    # Process the atlas slice-by-slice in Z.
    for z0 in tqdm(range(0, in_z, cube_size[0]), desc="Z-slices"):
        # For each Z-slice, iterate over Y and X.
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
                if len(batch_cubes) >= BATCH_SIZE:
                    flush_batch()
        # At the end of processing this Z-slice, flush any remaining cubes and wait for disk writes.
        if batch_cubes:
            flush_batch()
        logger.info(
            f"Finished Z-slice {z0} to {min(z0+cube_size[0], in_z)}. Waiting for disk writes...")
        write_queue.join()  # Wait until all queued items are written to disk.

    # Signal the writer thread to exit.
    write_queue.put(None)
    writer.join()
    logger.info(
        "Upsampling complete. Final upsampled atlas annotation saved in Zarr format.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your GPU settings.")
    main()
