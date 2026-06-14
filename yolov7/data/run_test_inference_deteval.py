#!/usr/bin/env python3
from pathlib import Path
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device


def main():
    weights = '/home/users/choij32/yolov7/runs/train/cholectrack20_yolov7_full/weights/best.pt'
    test_root = Path('/raid/cholectrack20_yolo/test_frames')
    out_root = Path('/raid/cholectrack20_predictions/yolov7_full_best')
    out_root.mkdir(parents=True, exist_ok=True)

    videos = ['VID01', 'VID06', 'VID07', 'VID12', 'VID25', 'VID39', 'VID111']

    imgsz = 640
    conf_thres = 0.001
    iou_thres = 0.65

    set_logging()
    device = select_device('0')
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    for vid in videos:
        src = test_root / vid
        if not src.exists():
            print(f'[skip] {vid}: source not found')
            continue

        dataset = LoadImages(str(src), img_size=imgsz, stride=stride)
        rows = []
        n_frames = 0

        for path, img, im0s, _ in dataset:
            n_frames += 1
            img_t = torch.from_numpy(img).to(device)
            img_t = img_t.float() / 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_t, augment=False)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

            frame_id = int(Path(path).stem)
            im0 = im0s.copy() if hasattr(im0s, 'copy') else im0s

            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det.tolist():
                        x1, y1, x2, y2 = xyxy
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        # DetEval TXT row: frame_id,unused,x,y,w,h,score,category_id
                        rows.append(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},{int(cls)}")

        out_file = out_root / f'{vid}.txt'
        out_file.write_text('\n'.join(rows))
        print(f'[save] {vid}: frames={n_frames}, detections={len(rows)} -> {out_file}')

    print('Done')


if __name__ == '__main__':
    main()
