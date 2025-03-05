import random
import torch
import os
import gdown
from PIL import Image, ImageDraw
import numpy as np
import argparse
from .util.slconfig import SLConfig
from .util.misc import nested_tensor_from_tensor_list
from .datasets import transforms as T
import warnings
from typing import Union

warnings.filterwarnings("ignore")

DEFAULT_CONF_THRESH = 0.23


def get_device():
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    return device


class ObjectCounter:
    """
    A class for counting objects in images based on text prompts and optional visual exemplars.
    """
    device = get_device()

    def __init__(self, model_path: str = "checkpoint_best_regular.pth",
                 config_path: str = "cfg_app.py",
                 crop_enabled: bool = True,
                 max_detections: int = 900):
        """
        Initializes the ObjectCounter with the model and configuration.

        Args:
            model_path: Path to the pretrained model checkpoint.
            config_path: Path to the configuration file.
            crop_enabled: Whether to enable adaptive cropping when maximum detections are reached.
            max_detections: The maximum count threshold before triggering adaptive cropping.
        """
        self.device = get_device()
        self.here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.here, config_path)
        model_path = os.path.join(self.here, model_path)
        self.annolid_git_repo = "https://github.com/healthonrails/annolid/releases/download/v1.2.0"
        self._REMOTE_MODEL_URL = f"{self.annolid_git_repo}/checkpoint_best_regular.pth"
        self._MD5 = "1492bfdd161ac1de471d0aafb32b174d"
        if not os.path.exists(model_path):
            gdown.cached_download(self._REMOTE_MODEL_URL,
                                  model_path,
                                  md5=self._MD5)

        self._REMOTE_BERT_MODEL_URL = f"{self.annolid_git_repo}/model.safetensors"
        self._BERT_MD5 = "cd18ceb6b110c04a8033ce01de41b0b7"
        self._BERT_MODEL_PATH = os.path.join(
            self.here, "checkpoints/bert-base-uncased/model.safetensors")
        if not os.path.exists(self._BERT_MODEL_PATH):
            gdown.cached_download(self._REMOTE_BERT_MODEL_URL,
                                  self._BERT_MODEL_PATH,
                                  md5=self._BERT_MD5)

        self._REMOTE_GROUNDINGDINO_MODEL_URL = f"{self.annolid_git_repo}/groundingdino_swinb_cogcoor.pth"
        self._GROUNDINGDINO_MD5 = "611367df01ee834e3baa408f54d31f02"
        self._GROUNDINGDINO_MODEL_PATH = os.path.join(
            self.here, "checkpoints/groundingdino_swinb_cogcoor.pth")
        if not os.path.exists(self._GROUNDINGDINO_MODEL_PATH):
            gdown.cached_download(self._REMOTE_GROUNDINGDINO_MODEL_URL,
                                  self._GROUNDINGDINO_MODEL_PATH,
                                  md5=self._GROUNDINGDINO_MD5)

        self.model = self._load_model(model_path, config_path, self.device)
        self.transform = self._build_transforms()
        self.crop_enabled = crop_enabled
        self.max_detections = max_detections

    def _build_transforms(self, size: int = 800, max_size: int = 1333) -> T.Compose:
        """Builds data transformations for image preprocessing."""
        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        return T.Compose([T.RandomResize([size], max_size=max_size), normalize])

    def _load_model(self, model_path: str, config_path: str, device: str) -> torch.nn.Module:
        """Loads the counting model from a checkpoint."""
        cfg = SLConfig.fromfile(config_path)
        cfg.merge_from_dict(
            {"text_encoder_type": os.path.join(self.here, "checkpoints/bert-base-uncased")})

        parser = argparse.ArgumentParser("Model Config")
        args = parser.parse_args()

        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            elif args_vars[k] != v:
                print(
                    f"Warning: Overriding config parameter '{k}' with command-line value.")

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        from .models.registry import MODULE_BUILD_FUNCS

        if args.modelname not in MODULE_BUILD_FUNCS._module_dict:
            raise ValueError(
                f"Model name '{args.modelname}' not found in registry.")
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build_func(args)

        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(checkpoint, strict=False)
        model.to(device).eval()
        return model

    def _get_box_inputs(self, prompts: list) -> list:
        """Extracts bounding box coordinates from prompt data."""
        return [
            [prompt[0], prompt[1], prompt[3], prompt[4]]
            for prompt in prompts
            if prompt[2] == 2.0 and prompt[5] == 3.0
        ]

    def _get_ind_to_filter(self, text: str, word_ids: list, keywords: str) -> list:
        """Determines the indices of word IDs to filter based on keywords."""
        if not keywords:
            return list(range(len(word_ids)))

        input_words = text.split()
        keywords_list = [keyword.strip() for keyword in keywords.split(",")]

        word_inds = []
        for keyword in keywords_list:
            try:
                start_index = 0 if not word_inds else word_inds[-1] + 1
                ind = input_words.index(keyword, start_index)
                word_inds.append(ind)
            except ValueError:
                raise ValueError(
                    f"Keyword '{keyword}' not found in input text: '{text}'")

        inds_to_filter = [ind for ind, word_id in enumerate(
            word_ids) if word_id in word_inds]
        return inds_to_filter

    def _convert_boxes_to_xyxy(self, image: Image.Image, boxes: np.ndarray) -> list:
        """Converts normalized bounding boxes to x1,y1,x2,y2 format."""
        h, w = image.height, image.width
        xyxy_boxes = []
        for box in boxes:
            center_x, center_y, box_w, box_h = box
            x1 = int((center_x - box_w / 2) * w)
            y1 = int((center_y - box_h / 2) * h)
            x2 = int((center_x + box_w / 2) * w)
            y2 = int((center_y + box_h / 2) * h)
            xyxy_boxes.append([x1, y1, x2, y2])
        return xyxy_boxes

    def _empty_nested_tensor(self):
        """
        Returns an empty NestedTensor to use in place of None.
        The dummy tensor shape here is (0, 3, 1, 1) which may be adjusted to match model expectations.
        """
        dummy = torch.empty((0, 3, 1, 1), device=self.device)
        return nested_tensor_from_tensor_list([dummy])

    def count_objects(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        exemplar_image: Union[str, Image.Image] = None,
        exemplar_boxes: list = None,
        confidence_threshold: float = DEFAULT_CONF_THRESH,
        keywords: str = "",
    ) -> list:
        """Counts objects and returns bounding boxes in x1,y1,x2,y2 format.

        Args:
            image: Path to the input image or a PIL Image object.
            text_prompt: Textual description of the object to count.
            exemplar_image: Path to the exemplar image or a PIL Image object (optional).
            exemplar_boxes: List of exemplar bounding boxes in normalized [xmin, ymin, xmax, ymax] format (optional).
            confidence_threshold: Confidence threshold for object detection (optional).
            keywords: Comma-separated keywords to filter detected objects (optional).

        Returns:
            A list of detected bounding boxes in x1,y1,x2,y2 format.
        """
        # Load the main image
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError(
                "image must be a file path (str) or a PIL Image object.")

        exemplar_prompts = {"image": None, "points": []}
        if exemplar_image:
            if isinstance(exemplar_image, str):
                exemplar_prompts["image"] = Image.open(
                    exemplar_image).convert("RGB")
            elif isinstance(exemplar_image, Image.Image):
                exemplar_prompts["image"] = exemplar_image.convert("RGB")
            else:
                raise ValueError(
                    "exemplar_image must be a file path (str) or a PIL Image object.")

            if exemplar_boxes:
                exemplar_prompts["points"] = [
                    [box[0], box[1], 2.0, box[2], box[3], 3.0] for box in exemplar_boxes
                ]

        # Preprocess the main image
        input_image, _ = self.transform(
            image_pil, {"exemplars": torch.tensor([])})
        input_image = input_image.unsqueeze(0).to(self.device)

        # Preprocess exemplar image if available
        input_image_exemplars = None
        exemplars_tensor = []
        if exemplar_prompts.get("image") is not None:
            exemplar_image_pil = exemplar_prompts["image"]
            input_image_exemplars, exemplars_transformed = self.transform(
                exemplar_image_pil, {"exemplars": torch.tensor(
                    self._get_box_inputs(exemplar_prompts.get("points", [])))}
            )
            input_image_exemplars = input_image_exemplars.unsqueeze(
                0).to(self.device)
            exemplars_tensor = [
                exemplars_transformed["exemplars"].to(self.device)]

        with torch.no_grad():
            model_output = self.model(
                nested_tensor_from_tensor_list(input_image),
                nested_tensor_from_tensor_list(
                    input_image_exemplars) if input_image_exemplars is not None else self._empty_nested_tensor(),
                exemplars_tensor if len(exemplars_tensor) > 0 else None,
                [torch.tensor([0]).to(self.device)] * len(input_image),
                captions=[text_prompt + " ."] * len(input_image),
            )

        ind_to_filter = self._get_ind_to_filter(
            text_prompt, model_output["token"][0].word_ids, keywords)
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]

        if keywords.strip():
            box_mask = (logits > confidence_threshold).sum(
                dim=-1) == len(ind_to_filter)
        else:
            box_mask = logits.max(dim=-1).values > confidence_threshold
        filtered_boxes = boxes[box_mask, :].cpu().numpy()

        # Adaptive cropping branch: if detections are >= max_detections, process by quadrants.
        if self.crop_enabled and (filtered_boxes.shape[0] >= self.max_detections):
            print(
                "Detected high number of objects, applying adaptive cropping (simple quadrant crop)...")
            width, height = image_pil.size

            # Convert exemplar_boxes from normalized to full-image absolute coordinates.
            exemplar_boxes_abs = []
            if exemplar_boxes:
                for box in exemplar_boxes:
                    ex_left = int(box[0] * width)
                    ex_top = int(box[1] * height)
                    ex_right = int(box[2] * width)
                    ex_bottom = int(box[3] * height)
                    exemplar_boxes_abs.append(
                        [ex_left, ex_top, ex_right, ex_bottom])

            # Define four quadrants: (left, top, right, bottom)
            quadrants = [
                (0, 0, width // 2, height // 2),
                (width // 2, 0, width, height // 2),
                (0, height // 2, width // 2, height),
                (width // 2, height // 2, width, height)
            ]
            all_boxes = []

            # Helper function to check if two boxes overlap.
            def boxes_overlap(box1, box2):
                # box format: [x1, y1, x2, y2]
                inter_left = max(box1[0], box2[0])
                inter_top = max(box1[1], box2[1])
                inter_right = min(box1[2], box2[2])
                inter_bottom = min(box1[3], box2[3])
                return inter_left < inter_right and inter_top < inter_bottom

            # Process each quadrant.
            for (left, top, right, bottom) in quadrants:
                crop_img = image_pil.crop((left, top, right, bottom))
                crop_w = right - left
                crop_h = bottom - top
                crop_exemplar_boxes = []  # to hold normalized exemplar boxes for this crop

                # Arrange exemplar patches in a grid (starting at top-left of the crop).
                current_x = 0
                current_y = 0
                max_row_height = 0
                spacing = 5  # spacing between patches (in pixels)
                if exemplar_boxes_abs:
                    for box in exemplar_boxes_abs:
                        # Extract the exemplar patch from the original full image.
                        patch = image_pil.crop(
                            (box[0], box[1], box[2], box[3]))
                        patch_width, patch_height = patch.size

                        # If the patch does not fit in the current row, move to the next row.
                        if current_x + patch_width > crop_w:
                            current_x = 0
                            current_y += max_row_height + spacing
                            max_row_height = 0

                        # If the patch does not fit vertically in the crop, skip it.
                        if current_y + patch_height > crop_h:
                            continue

                        # Paste the patch onto the crop image at the computed (current_x, current_y).
                        crop_img.paste(patch, (current_x, current_y))

                        # Save normalized coordinates of this pasted patch (for the transform).
                        norm_x1 = current_x / crop_w
                        norm_y1 = current_y / crop_h
                        norm_x2 = (current_x + patch_width) / crop_w
                        norm_y2 = (current_y + patch_height) / crop_h
                        crop_exemplar_boxes.append(
                            [norm_x1, norm_y1, norm_x2, norm_y2])

                        # Update x position and row height for the grid.
                        current_x += patch_width + spacing
                        max_row_height = max(max_row_height, patch_height)

                # Now pass the normalized exemplar boxes into the transform.
                exemplar_tensor = torch.tensor(
                    crop_exemplar_boxes) if crop_exemplar_boxes else torch.tensor([])
                crop_tensor, _ = self.transform(
                    crop_img, {"exemplars": exemplar_tensor})
                crop_tensor = crop_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if input_image_exemplars is not None and len(exemplars_tensor) > 0:
                        replicated_exemplars = [
                            exemplars_tensor[0]] * len(crop_tensor)
                        exemplar_input = nested_tensor_from_tensor_list(
                            input_image_exemplars)
                    else:
                        replicated_exemplars = None
                        exemplar_input = self._empty_nested_tensor()
                    crop_output = self.model(
                        nested_tensor_from_tensor_list(crop_tensor),
                        exemplar_input,
                        replicated_exemplars,
                        [torch.tensor([0]).to(self.device)] * len(crop_tensor),
                        captions=[text_prompt + " ."] * len(crop_tensor),
                    )

                crop_logits = crop_output["pred_logits"].sigmoid()[
                    0][:, ind_to_filter]
                crop_boxes = crop_output["pred_boxes"][0]
                if keywords.strip():
                    crop_box_mask = (crop_logits > confidence_threshold).sum(
                        dim=-1) == len(ind_to_filter)
                else:
                    crop_box_mask = crop_logits.max(
                        dim=-1).values > confidence_threshold
                crop_filtered_boxes = crop_boxes[crop_box_mask, :].cpu(
                ).numpy()

                # Adjust detection boxes from crop coordinates to full image coordinates.
                for box in crop_filtered_boxes:
                    center_x = box[0] * crop_w + left
                    center_y = box[1] * crop_h + top
                    box_w = box[2] * crop_w
                    box_h = box[3] * crop_h
                    x1 = int(center_x - box_w / 2)
                    y1 = int(center_y - box_h / 2)
                    x2 = int(center_x + box_w / 2)
                    y2 = int(center_y + box_h / 2)
                    det_box = [x1, y1, x2, y2]

                    # Check if this detected box overlaps with any exemplar box.
                    overlap = False
                    for ex_box in exemplar_boxes_abs:
                        if boxes_overlap(det_box, ex_box):
                            overlap = True
                            break
                    if not overlap:
                        all_boxes.append(det_box)
            # Deduplicate detection boxes.
            unique_boxes = [list(x) for x in set(tuple(b) for b in all_boxes)]
            # Remove any detection box that overlaps with any exemplar box.
            final_boxes = []
            for det_box in unique_boxes:
                overlap = False
                for ex_box in exemplar_boxes_abs:
                    if boxes_overlap(det_box, ex_box):
                        overlap = True
                        break
                if not overlap:
                    final_boxes.append(det_box)
            print(
                f"Adaptive cropping complete, detected {len(final_boxes)} objects.")
            return final_boxes
        else:
            return self._convert_boxes_to_xyxy(image_pil, filtered_boxes)


if __name__ == "__main__":
    input_image_path = "strawberry.jpg"
    text_prompt = "blueberries"
    exemplar_image_path = "strawberry.jpg"
    exemplar_boxes = [[0.1, 0.1, 0.2, 0.2]]  # normalized coordinates

    pretrain_model_path = "checkpoint_best_regular.pth"
    config_path = "cfg_app.py"

    # Initialize the object counter with adaptive cropping enabled.
    object_counter = ObjectCounter(
        pretrain_model_path, config_path=config_path, crop_enabled=True, max_detections=900)

    # Get bounding boxes using image path
    detected_boxes_path = object_counter.count_objects(
        input_image_path,
        text_prompt,
        exemplar_image=exemplar_image_path,
        exemplar_boxes=exemplar_boxes,
        confidence_threshold=0.3,
        keywords="blueberries",
    )
    print("Detected Boxes (path input):", detected_boxes_path)

    # Get bounding boxes using PIL Image object
    image_pil = Image.open(input_image_path)
    exemplar_image_pil = Image.open(exemplar_image_path)
    detected_boxes_pil = object_counter.count_objects(
        image_pil,
        text_prompt,
        exemplar_image=exemplar_image_pil,
        exemplar_boxes=exemplar_boxes,
        confidence_threshold=0.3,
        keywords="blueberries",
    )
    print("Detected Boxes (PIL input):", detected_boxes_pil)
