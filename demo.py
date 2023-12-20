
from OpenMineChangeDetection.utils import download_prep_oms2cd
from OpenMineChangeDetection.datasets import OMS2CD
import rasterio
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import os
from skimage.exposure import equalize_hist, equalize_adapthist
from tqdm import tqdm
import glob
import itertools

def normalize_histogram(image):
    """
    Apply histogram normalization to an image.
    Assumes image is a NumPy array with shape (channels, height, width).
    """
    # Normalize each channel separately
    for i in range(image.shape[0]):
        image[i, :, :] = equalize_hist(image[i, :, :]) * 255
    return image

from skimage.exposure import equalize_adapthist

def adaptive_histogram_equalization(image):
    """
    Apply adaptive histogram equalization (CLAHE) to an image.
    Assumes image is a NumPy array with shape (channels, height, width).
    """
    for i in range(image.shape[0]):
        image[i, :, :] = equalize_adapthist(image[i, :, :], clip_limit=0.05) * 255
    return image.astype(np.uint8)

def apply_mask_to_image(image, mask, mask_alpha_channel):
    # Convert RGB image to RGBA by adding an opaque alpha channel
    alpha_channel = np.full(shape=image.shape[1:], fill_value=255, dtype=np.uint8)
    image_rgba = np.concatenate((image, alpha_channel[np.newaxis, :, :]), axis=0)

    # Normalize alpha to [0, 1]
    normalized_alpha = mask_alpha_channel / 255.0

    # Vectorized operation for alpha blending
    image_rgba[:3, :, :] = normalized_alpha * mask[:3, :, :] + (1 - normalized_alpha) * image_rgba[:3, :, :]

    return image_rgba

def find_bounding_box(mask, p=0.1):
    """Find the bounding box of the white areas in the mask and add a margin."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Calculate the margin size
    height, width = mask.shape
    x_margin = int((xmax - xmin) * p)
    y_margin = int((ymax - ymin) * p)

    # Expand the bounding box with the margin, ensuring it stays within the image boundaries
    xmin = max(xmin - x_margin, 0)
    xmax = min(xmax + x_margin, width - 1)
    ymin = max(ymin - y_margin, 0)
    ymax = min(ymax + y_margin, height - 1)

    return xmin, ymin, xmax, ymax

def crop_image(image, bounding_box):
    """Crop the image to the specified bounding box."""
    xmin, ymin, xmax, ymax = bounding_box
    return image[:, ymin:ymax+1, xmin:xmax+1]

def adjust_color_intensity(color):
    """ Adjust the color intensity to have a similar perceived brightness. """
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    target_luminance = 0.299 * 255 + 0.587 * 255  # Luminance of yellow
    if luminance < target_luminance:
        scale_factor = target_luminance / luminance
        new_color = [min(255, int(c * scale_factor)) for c in color]
        return new_color
    else:
        return color

def create_hybrid_overlay(gt_mask, predicted_mask, alpha=0.4):
    # TP = Yellow
    # FN = Red
    # FP = Blue

    # Convert masks from 0-255 to binary (0 or 1)
    gt_binary = gt_mask > 0
    predicted_binary = predicted_mask > 0

    # Calculate TP, FN, and FP
    TP = np.logical_and(gt_binary, predicted_binary)  # True Positives
    FN = np.logical_and(gt_binary, np.logical_not(predicted_binary))  # False Negatives
    FP = np.logical_and(np.logical_not(gt_binary), predicted_binary)  # False Positives

    # Create an empty RGBA image for the overlay
    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.uint8)

    yellow = [255, 255, 0]
    red = adjust_color_intensity([255, 0, 0])
    blue = adjust_color_intensity([0, 0, 255])

    overlay[TP] = yellow + [int(255 * alpha)]  # Yellow with alpha
    overlay[FN] = red + [int(255 * alpha)]    # Adjusted Red with alpha
    overlay[FP] = blue + [int(255 * alpha)]   # Adjusted Blue with alpha


    # Flip back to [c, h, w]
    overlay = overlay.transpose((2, 0, 1))

    return overlay


def create_gif(image_path1, image_path2, mask_path, area_mask_path, gt_mask_path, \
               output_gif_path, alpha=0.4, equalize=False, loops=2):
    logging.basicConfig(level=logging.INFO)

    if True:
        # Read the GeoTIFFs
        try:
            with rasterio.open(image_path1) as src1:
                image1 = src1.read()
            with rasterio.open(image_path2) as src2:
                image2 = src2.read()
            with rasterio.open(mask_path) as mask_src:
                mask = mask_src.read()
        except Exception as e:
            logging.error(f'Error while loading one of the .tif files: {e}')
            return

        if area_mask_path:
            try:
                with rasterio.open(area_mask_path) as mask_src:
                    area_mask = mask_src.read(1)  # Assuming the mask is single-band
            except Exception as e:
                logging.error(f'Error loading area mask: {e}')
                return

            # Find bounding box and crop images
            bounding_box = find_bounding_box(area_mask)
            image1 = crop_image(image1, bounding_box)
            image2 = crop_image(image2, bounding_box)
            mask = crop_image(mask, bounding_box)

        if equalize:
            image1 = adaptive_histogram_equalization(image1)
            image2 = adaptive_histogram_equalization(image2)

        # Normalize and convert to RGB (if needed)
        # This part depends on your data format

        # Prepare the mask
        mask = mask.astype(np.uint8)

        if gt_mask_path:
            try:
                with rasterio.open(gt_mask_path) as gt_mask_src:
                    gt_mask = gt_mask_src.read()
            except Exception as e:
                logging.error(f'Error loading GT mask: {e}')

            if area_mask_path:
                gt_mask = crop_image(gt_mask, bounding_box)
            # Create the hybrid overlay
            mask_overlay = create_hybrid_overlay(gt_mask[0], mask[0], alpha)
            alpha_channel = np.full(gt_mask.shape[1:], int(255 * alpha), dtype=np.uint8)

        else:
            # mask_rgb = np.repeat(mask, 3, axis=0)
            # Define a color for the mask (yellow: full red, full green, no blue)
            color = np.array([1, 1, 0], dtype=np.uint8)  # Yellow color

            # Apply the color to the mask
            mask_rgb = np.stack([mask[0, :, :] * color[i] for i in range(3)], axis=0)
            alpha_channel = np.full(mask.shape[1:], int(255 * alpha), dtype=np.uint8)
            mask_overlay = np.concatenate((mask_rgb, alpha_channel[np.newaxis, :, :]), axis=0)

        # This part is used to remove the black background from the mask :)
        nz = mask_overlay == 0
        mask_overlay[nz][3] = 0
        alpha_channel[nz[3]] = 0

        frames = []
        for _ in range(loops):  # Number of loops
            # Apply mask to images
            image1_rgba = apply_mask_to_image(image1, mask_overlay, alpha_channel)
            image2_rgba = apply_mask_to_image(image2, mask_overlay, alpha_channel)

            # Convert to PIL Image and add to frames
            frame1 = Image.fromarray(image1_rgba.transpose((1, 2, 0)))
            frames.append(frame1)

            frame2 = Image.fromarray(image2_rgba.transpose((1, 2, 0)))
            frames.append(frame2)

        # Save as a GIF
        try:
            frames[0].save(output_gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=1)
        except Exception as e:
            logging.error(f'Error saving .gif file: {e}')
            return
        logging.info(f"GIF created successfully as {output_gif_path}")
    # except Exception as e:
    #     logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    OMS2CD_PATH = Path('/data/OMS2CD/')
    SITE_DATA = Path('/work/OpenMineChangeDetection/site_data/')
    OUTPUTS = Path('/outputs/')

    download_prep_oms2cd(output_dir=str(OMS2CD_PATH))
    
    # split = 'val' # 'test'
    # validated_or_not = 'validated' # 'not_validated'
    # threshold = '0.4' # '0.6'
    # model = 'lsnet' # 'ddpmcd', 'tinycd'

    # Define the options for each variable
    splits = ['val', 'test']
    validated_options = ['validated', 'not_validated']
    thresholds = ['0.4', '0.6']
    models = ['lsnet', 'ddpmcd', 'tinycd']

    # Use itertools.product to create all combinations
    for split, validated_or_not, threshold, model in tqdm(itertools.product(splits, validated_options, thresholds, models)):
        this_out_dir = OUTPUTS / os.path.join(validated_or_not, threshold, model)

        logging.info(f'Creating directory path {this_out_dir}.')
        os.makedirs(this_out_dir, mode=0o777, exist_ok=True)

        valid_ext = '_validated' if validated_or_not == 'validated' else ''

        if validated_or_not == 'validated':
            dataset = OMS2CD(root=OMS2CD_PATH, split=split, load_area_mask=True)
        else:
            # TODO pull case study data from GDrive
            continue

        facilities = dataset.get_facilities()
        for facility in sorted(facilities):
            facility_file_indices = dataset.get_facility_file_indices(facility)
            mask_list = []
            date_list = []
            cumulative_mask = None  # Initialize the cumulative mask

            area_mask_path = os.path.join(dataset.root_dir, 'area_mask', f'{facility}.tif')
            if os.path.isfile(area_mask_path):
                pass

            date_range_list = []  # This will store the date ranges for each file index
            for file_index in facility_file_indices:
                file_info = dataset.file_list[file_index]
                predate, postdate = file_info[-3:-1]
                date_range_list.append(f"{predate} - {postdate}")

                src_img1 = file_info[0]
                src_img2 = file_info[1]

                # logging.info(src_img1)
                # logging.info(src_img2)

                gt_mask = file_info[2] # Ground truth file
                area_mask = file_info[3] # area mask file

                pred_img = f'pred_{facility}_*_{predate}_{postdate}.tif'
                pred_dir = SITE_DATA / os.path.join(f'modelchanges{valid_ext}', threshold, f'masks_{model}')

                pattern = pred_dir / f'pred_{facility}_*_{predate}_{postdate}.tif'
                # logging.info(f'Looking for files in {pattern}')
                matching_files = glob.glob(str(pattern))
                if len(matching_files) == 0:
                    logging.warning(f'File for the predicted image not found at {pattern}')
                    continue

                pred_img = matching_files[0]

                suffix = ''
                out_img = f'{facility}_{predate}_{postdate}_{suffix}.gif'

                # logging.info()
                # break

                create_gif(OMS2CD_PATH / src_img1, \
                        OMS2CD_PATH / src_img2, \
                        pred_dir / pred_img, \
                        OMS2CD_PATH / os.path.join('area_mask', area_mask), \
                        OMS2CD_PATH / os.path.join('mask', gt_mask), \
                        this_out_dir / out_img, \
                        alpha=0.5, \
                        equalize=True, \
                        loops=3)

    # src_img1 = 's2_Werris Creek_150.633355555961_-31.3853783732025_2019-03-01.tif'
    # src_img2 = 's2_Werris Creek_150.633355555961_-31.3853783732025_2019-04-01.tif'

    # pred_img = 'pred_Werris Creek_72_2019-03-01_2019-04-01.tif'

    # out_img = 'pred_Werris Creek_72_2019-03-01_2019-04-01_validated_cropped.gif'

    # area_mask = 'Werris Creek.tif'
    # gt_mask = 'Werris Creek_0218.tif'

    # # Usage
    # create_gif(OMS2CD_PATH / src_img1, \
    #         OMS2CD_PATH / src_img2, \
    #         SITE_DATA / os.path.join(f'modelchanges{valid_ext}', threshold, f'masks_{model}', str(pred_img)), \
    #         OMS2CD_PATH / os.path.join('area_mask', area_mask), \
    #         OMS2CD_PATH / os.path.join('mask', gt_mask), \
    #         this_out_dir / out_img, \
    #         alpha=0.5, \
    #         equalize=True, \
    #         loops=3)
