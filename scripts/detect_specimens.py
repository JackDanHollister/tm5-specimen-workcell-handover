"""Run the bundled GUI YOLO model on one saved overhead photograph."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drawer',default='drawer_001');p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/specimens.json');args=p.parse_args()
    from ultralytics import YOLO
    image=ROOT/'data/drawers'/args.drawer/'eih/image.png'
    model=YOLO(str(ROOT/'models/yolo/mk2_best_model.pt'))
    prediction=model.predict(str(image),device=args.device,verbose=False)[0]
    rows=[]
    for box in prediction.boxes:
        name=prediction.names[int(box.cls)]
        if name!='specimen':continue
        xy=box.xyxy[0].cpu().tolist()
        rows.append({'id':len(rows)+1,'bbox_xyxy_pixels':xy,'centroid_xy_pixels':[(xy[0]+xy[2])/2,(xy[1]+xy[3])/2],'confidence':float(box.conf)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'drawer':args.drawer,'image':str(image.relative_to(ROOT)),
        'detections':rows,'coordinate_frame':'EIH original pixels','pin_centres_measured':False,
        'next_step':'Register depth into EIH pixels; use supported surface/framing estimates, then recover the handling-pin axis'},indent=2)+'\n')
    print(f'{len(rows)} specimen framing detections saved')


if __name__=='__main__':main()
