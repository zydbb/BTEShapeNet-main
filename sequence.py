import os
import argparse
import torch
import numpy as np
from PIL import Image
from model.BTEShapeNet import BTEShapeNet
import model.Config as config
from tqdm import tqdm
import subprocess

from torchvision import transforms

def load_model(model_path):
    cfg = config.get_BTEShape_config()
    model = BTEShapeNet(cfg, mode='test', deepsuper=True)
    state = torch.load(model_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def load_images(image_dir):
    image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
    images = []
    for fname in image_list:
        img = Image.open(os.path.join(image_dir, fname)).convert('L')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        images.append((fname, img_tensor))
    return images

def predict_sequence(model, images, threshold):
    results = []
    with torch.no_grad():
        for idx, (fname, img) in enumerate(tqdm(images)):
            pred = model(img)
            pred = pred.squeeze().numpy()
            mask = (pred > threshold).astype(np.uint8)
            coords = np.argwhere(mask > 0)
            if coords.size == 0:
                results.append((idx, []))
            else:
                cx, cy = coords.mean(axis=0).astype(int)
                results.append((idx, [(1, cx, cy)]))  # object_id=1
    return results

def save_to_txt(results, save_path):
    with open(save_path, 'w') as f:
        f.write(f"1 data1 {len(results)} 1\n")
        for frame_idx, objs in results:
            if not objs:
                f.write(f"frame:{frame_idx} 0\n")
            else:
                x, y = objs[0][1], objs[0][2]
                f.write(f"frame:{frame_idx} 1 object:1 {x} {y}\n")

def run_score(score_script, gt_path, pred_path):
    if not os.path.exists(score_script):
        print(f"Error: score script not found at {score_script}. Please check the path.")
        return
    cmd = f"python {score_script} {gt_path} {pred_path}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='datasets/data4', help='Path to sequence .bmp images')
    parser.add_argument('--model_path', type=str, default='log/NUDT-SIRST/BTEShapeNet_NUDT-SIRST_best.pth.tar', help='Path to .pth model checkpoint')
    parser.add_argument('--gt_path', type=str, default='datasets/data_label/data4.txt', help='Path to GT data1 file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--pred_path', type=str, default='prediction.txt', help='Output prediction txt path')
    parser.add_argument('--score_script', type=str, default='datasets/评分程序python版/python/main.py', help='Path to official score.py script')
    args = parser.parse_args()
    print("\n[1] Loading model...")
    model = load_model(args.model_path)

    print("[2] Loading images...")
    images = load_images(args.image_dir)

    print("[3] Running inference...")
    results = predict_sequence(model, images, args.threshold)

    print("[4] Saving predictions...")
    save_to_txt(results, args.pred_path)

    print("[5] Scoring with official script...")
    run_score(args.score_script, args.gt_path, args.pred_path)


if __name__ == '__main__':
    main()