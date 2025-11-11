#!/usr/bin/env python3
"""
Batch mask creation script using SAM2.
Shows one reference image for user to mark the object, then automatically segments all images.
"""

import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path
import torch
from tqdm import tqdm

try:
    from sam2.sam2.build_sam import build_sam2
    from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Please install SAM2 to use this script.")


class MaskCreator:
    def __init__(self, checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt", 
                 model_cfg="./sam2/configs/sam2.1/sam2.1_hiera_l.yaml"):
        """Initialize SAM2 predictor."""
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not available. Please install it first.")
        
        print("Loading SAM2 model...")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.point_coords = []
        self.point_labels = []
        self.box = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding box or clicking points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Ctrl+Click: Add foreground point
                self.point_coords.append([x, y])
                self.point_labels.append(1)
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', self.display_image)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                # Shift+Click: Add background point
                self.point_coords.append([x, y])
                self.point_labels.append(0)
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', self.display_image)
            else:
                # Start drawing box
                self.drawing = True
                self.start_point = (x, y)
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                temp_image = self.display_image.copy()
                if self.start_point and self.end_point:
                    cv2.rectangle(temp_image, self.start_point, self.end_point, (255, 0, 0), 2)
                cv2.imshow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                if self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    self.box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                    cv2.rectangle(self.display_image, self.start_point, self.end_point, (255, 0, 0), 2)
                    cv2.imshow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', self.display_image)
    
    def get_prompt_from_reference(self, image_path):
        """Show reference image and get user prompt (box or points)."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = image.copy()
        self.point_coords = []
        self.point_labels = []
        self.box = None
        
        cv2.namedWindow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', self.mouse_callback)
        
        instructions = [
            "Instructions:",
            "1. Drag mouse to draw bounding box around object",
            "OR",
            "2. Ctrl+Click on object (green = foreground)",
            "3. Shift+Click on background (red = background)",
            "4. Press ENTER to confirm, ESC to cancel"
        ]
        
        # Add instructions as overlay
        overlay = image.copy()
        y_offset = 30
        for line in instructions:
            cv2.putText(overlay, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25
        
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, self.display_image)
        cv2.imshow('Reference Image - Click object (Ctrl+Click) or drag box (Drag)', self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if self.box is not None or len(self.point_coords) > 0:
                    break
                else:
                    print("Please draw a box or click points first!")
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None
        
        cv2.destroyAllWindows()
        
        # Convert to numpy arrays
        if self.box is not None:
            box = self.box.astype(np.float32)
            return box, None
        elif len(self.point_coords) > 0:
            coords = np.array(self.point_coords, dtype=np.float32)
            labels = np.array(self.point_labels, dtype=np.int32)
            return None, (coords, labels)
        else:
            return None, None
    
    def segment_image(self, image, box=None, point_coords=None, point_labels=None):
        """Segment a single image using SAM2."""
        if image is None:
            return None
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_rgb)
            
            if box is not None:
                # Use bounding box
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],  # Add batch dimension
                    multimask_output=False,
                )
            elif point_coords is not None:
                # Use point prompts
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords[None, :, :],  # Add batch dimension
                    point_labels=point_labels[None, :],
                    box=None,
                    multimask_output=False,
                )
            else:
                return None
            
            mask = masks[0].astype(np.bool_)
        
        return mask
    
    def process_all_images(self, image_dir, output_dir=None, reference_idx=0, 
                          box=None, point_prompt=None, reference_image_path=None):
        """Process all images in directory."""
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} images")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(image_dir, 'masks')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load reference image to get dimensions and convert point prompts to box if needed
        ref_image = None
        ref_h, ref_w = None, None
        if reference_image_path is not None:
            ref_image = cv2.imread(reference_image_path)
            if ref_image is not None:
                ref_h, ref_w = ref_image.shape[:2]
        
        # If we have point prompts, first segment reference image to get a bounding box
        # This makes the box robust to different image sizes
        if point_prompt is not None and ref_image is not None:
            print("Segmenting reference image to get bounding box...")
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(ref_image_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_prompt[0][None, :, :],
                    point_labels=point_prompt[1][None, :],
                    box=None,
                    multimask_output=False,
                )
                ref_mask = masks[0].astype(np.bool_)
                
                # Convert mask to bounding box
                coords = np.where(ref_mask > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    # Add some padding (5%)
                    padding_x = int((x_max - x_min) * 0.05)
                    padding_y = int((y_max - y_min) * 0.05)
                    box = np.array([
                        max(0, x_min - padding_x),
                        max(0, y_min - padding_y),
                        min(ref_w, x_max + padding_x),
                        min(ref_h, y_max + padding_y)
                    ], dtype=np.float32)
                    print(f"Converted point prompts to bounding box: {box}")
                    point_prompt = None  # Use box instead
        
        # Process all images
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            image = cv2.imread(image_path)
            if image is None:
                print(f"\nWarning: Could not load {image_path}, skipping...")
                continue
            
            # For box prompts, scale box if image size is different from reference
            current_box = box
            if box is not None and ref_h is not None and ref_w is not None:
                curr_h, curr_w = image.shape[:2]
                if ref_h != curr_h or ref_w != curr_w:
                    # Scale box coordinates
                    scale_x = curr_w / ref_w
                    scale_y = curr_h / ref_h
                    current_box = box.copy()
                    current_box[0] *= scale_x
                    current_box[1] *= scale_y
                    current_box[2] *= scale_x
                    current_box[3] *= scale_y
            
            # Segment image
            mask = self.segment_image(image, box=current_box, 
                                     point_coords=point_prompt[0] if point_prompt else None,
                                     point_labels=point_prompt[1] if point_prompt else None)
            
            if mask is None:
                print(f"\nWarning: Segmentation failed for {os.path.basename(image_path)}, skipping...")
                continue
            
            # Save mask with same name as color file
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(output_dir, f"{image_name}.png")
            
            # Save as binary mask (white object on black background)
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
        
        print(f"\nDone! Masks saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch mask creation using SAM2")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing color images")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for masks (default: image_dir/masks)")
    parser.add_argument("--reference_idx", type=int, default=0,
                       help="Index of reference image for prompting (default: 0)")
    parser.add_argument("--checkpoint", type=str, 
                       default="./sam2/checkpoints/sam2.1_hiera_large.pt",
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str,
                       default="./sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                       help="Path to SAM2 model config")
    
    args = parser.parse_args()
    
    if not SAM2_AVAILABLE:
        print("Error: SAM2 is not available. Please install it first.")
        return
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.image_dir}")
        return
    
    image_files = sorted(image_files)
    
    if args.reference_idx >= len(image_files):
        print(f"Error: Reference index {args.reference_idx} is out of range (0-{len(image_files)-1})")
        return
    
    # Initialize mask creator
    creator = MaskCreator(checkpoint=args.checkpoint, model_cfg=args.model_cfg)
    
    # Get prompt from reference image
    print(f"\nShowing reference image: {os.path.basename(image_files[args.reference_idx])}")
    print("Mark the object in this image:")
    box, point_prompt = creator.get_prompt_from_reference(image_files[args.reference_idx])
    
    if box is None and point_prompt is None:
        print("No prompt provided. Exiting.")
        return
    
    # Process all images
    print("\nProcessing all images...")
    creator.process_all_images(args.image_dir, args.output_dir, args.reference_idx,
                              box=box, point_prompt=point_prompt, 
                              reference_image_path=image_files[args.reference_idx])
    
    print("\nBatch mask creation complete!")


if __name__ == "__main__":
    main()

