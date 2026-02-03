#!/usr/bin/env python3
#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

"""
matcher.py
Moderner Bildmatcher mit CNN-Embeddings + FAISS.

Usage:
  # 1) Index bauen
  python matcher.py build --ref images/reference --outmodel model/

  # 2) Bild suchen
  python matcher.py query --model model/ --img images/query/test1.jpg --topk 6
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import faiss
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import cv2
import numpy as np

def _local_match_score(query_path: str, cand_path: str) -> float:
    """
    Return a 0..1 score using ORB keypoints + BF matching + RANSAC inliers.
    Higher is better. Robust to small pose/light changes.
    """
    q = cv2.imread(query_path)
    c = cv2.imread(cand_path)
    if q is None or c is None: 
        return 0.0

    # convert to grayscale and lightly equalize (contrast)
    qg = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    qg = cv2.createCLAHE(2.0, (8,8)).apply(qg)
    cg = cv2.createCLAHE(2.0, (8,8)).apply(cg)

    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    kq, dq = orb.detectAndCompute(qg, None)
    kc, dc = orb.detectAndCompute(cg, None)
    if dq is None or dc is None or len(dq) < 8 or len(dc) < 8:
        return 0.0

    # brute-force Hamming matcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(dq, dc, k=2)

    # Lowe's ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return 0.0

    src_pts = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kc[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # RANSAC homography to count geometric inliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = int(mask.sum())

    # normalize by keypoints to get ~0..1
    denom = max(len(kq), len(kc), 1)
    score = min(inliers / denom, 1.0)
    return float(score)


def rerank_with_local(query_path: str, results, alpha: float = 0.6):
    """
    Combine CNN score (already in results) with local ORB/RANSAC score.
    results: list of (candidate_abs_path, cnn_score)
    alpha: weight for CNN (0..1). (1-alpha) for local
    Returns a new list sorted by combined score desc.
    """
    # rescale CNN scores ~[0..1] in case they are not already
    if results:
        cnns = np.array([r[1] for r in results], dtype=np.float32)
        # avoid degenerate scaling
        lo, hi = float(cnns.min()), float(cnns.max())
        if hi > lo:
            cnns_n = (cnns - lo) / (hi - lo)
        else:
            cnns_n = np.ones_like(cnns) * 0.5
    else:
        return results

    combined = []
    for (i, (path, s_cnn)) in enumerate(results):
        s_local = _local_match_score(query_path, path)
        s = alpha * float(cnns_n[i]) + (1.0 - alpha) * float(s_local)
        combined.append((path, s, s_cnn, s_local))

    # sort by combined score desc and return same shape as input: (path, combined_score)
    combined.sort(key=lambda x: x[1], reverse=True)
    return [(p, float(s)) for (p, s, _, _) in combined]

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    return sorted([p for p in Path(folder).glob('*') if p.suffix.lower() in exts])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img
def underwater_preprocess_pil(pil_img):
    """Simple color correction + CLAHE for underwater photos; returns PIL RGB."""
    img = np.array(pil_img)[:, :, ::-1]  # PIL RGB -> BGR for OpenCV

    # Gray-world white balance
    b, g, r = cv2.split(img.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b *= (mean_gray / (mean_b + 1e-6)); g *= (mean_gray / (mean_g + 1e-6)); r *= (mean_gray / (mean_r + 1e-6))
    img = cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)

    # CLAHE on L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    img = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    # back to PIL RGB
    return Image.fromarray(img[:, :, ::-1])

from torchvision.models import resnet50, ResNet50_Weights

def get_model(device):
    weights = ResNet50_Weights.DEFAULT
    base = resnet50(weights=weights)
    base.eval()
    feature_extractor = torch.nn.Sequential(*list(base.children())[:-1]).to(device)
    return feature_extractor


def _resize_pad(pil_img, size=256):
    """Keep aspect ratio; pad to square (size x size)."""
    w, h = pil_img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0,0,0))
    canvas.paste(img, ((size - new_w)//2, (size - new_h)//2))
    return canvas

def extract_embedding(pil_img, model, device, tta=True):
    # 1) Underwater correction
    pil_img = underwater_preprocess_pil(pil_img)
    # 2) Preserve full content (no aggressive crop)
    pil_img = _resize_pad(pil_img, size=256)

    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    if tta:
        # Five crops (224) + flips → robust embedding
        five = transforms.FiveCrop(224)(pil_img)
        views = list(five) + [img.transpose(Image.FLIP_LEFT_RIGHT) for img in five]
    else:
        views = [transforms.CenterCrop(224)(pil_img)]

    embs = []
    with torch.no_grad():
        for v in views:
            x = base_tf(v).unsqueeze(0).to(device)
            e = model(x).squeeze().cpu().numpy()
            e = e / (np.linalg.norm(e) + 1e-12)
            embs.append(e.astype('float32'))

    emb = np.mean(embs, axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype('float32')


def build_index(reference_folder, out_model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    ref_paths = list_images(reference_folder)
    if not ref_paths:
        raise SystemExit("Keine Referenzbilder gefunden.")

    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    image_names = []

    print("Berechne Embeddings der Referenzbilder...")
    for p in tqdm(ref_paths):
        img = load_image(p)
        emb = extract_embedding(img, model, device)
        embeddings.append(emb)
        image_names.append(str(p.resolve()))

    feats = np.vstack(embeddings).astype('float32')

    print("Erstelle FAISS-Index...")
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)  
    index.add(feats)

    print("Speichere Modelle...")
    faiss.write_index(index, str(out_model_dir / "faiss.index"))
    with open(out_model_dir / "meta.json", "w", encoding='utf8') as f:
        json.dump({"image_names": image_names}, f, ensure_ascii=False, indent=2)

    print("Fertig. Modell gespeichert in:", out_model_dir)

def query_image(model_dir, query_img_path, topk=6, show=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model_dir = Path(model_dir)

    index = faiss.read_index(str(model_dir / "faiss.index"))
    with open(model_dir / "meta.json", "r", encoding='utf8') as f:
        meta = json.load(f)
    image_names = meta["image_names"]

    img = load_image(query_img_path)
    emb_q = extract_embedding(img, model, device).astype('float32').reshape(1, -1)

    D, I = index.search(emb_q, topk)
    results = [(image_names[idx], float(dist)) for dist, idx in zip(D[0], I[0])] 
    results = rerank_with_local(query_img_path, results, alpha=0.6)

    if show:
        n = len(results) + 1
        plt.figure(figsize=(3*n, 4))
        plt.subplot(1, n, 1)
        plt.imshow(img)
        plt.title("Query")
        plt.axis('off')
        for i, (fname, dist) in enumerate(results, start=2):
            res_img = Image.open(fname).convert("RGB")
            plt.subplot(1, n, i)
            plt.imshow(res_img)
            plt.title(f"#{i-1}\n{Path(fname).name}\nScore={dist:.3f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return results

def main():
    parser = argparse.ArgumentParser(description="Moderner Bild-Matcher mit CNN-Embeddings + FAISS")
    sub = parser.add_subparsers(dest='cmd')

    p_build = sub.add_parser('build', help='Index aus Referenzbildern bauen')
    p_build.add_argument('--ref', required=True, help='Ordner mit Referenzbildern')
    p_build.add_argument('--outmodel', required=True, help='Ausgabe-Ordner für Modell')

    p_query = sub.add_parser('query', help='Bild suchen')
    p_query.add_argument('--model', required=True, help='Modell-Ordner')
    p_query.add_argument('--img', required=True, help='Suchbild')
    p_query.add_argument('--topk', type=int, default=5, help='Anzahl der Treffer')

    args = parser.parse_args()

    if args.cmd == 'build':
        build_index(args.ref, args.outmodel)
    elif args.cmd == 'query':
        query_image(args.model, args.img, args.topk)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()



