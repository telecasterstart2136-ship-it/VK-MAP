# 1. Google Drive をマウント
from google.colab import drive

drive.mount("/content/drive")

import os
import pickle
import faiss
import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 2. パスと設定
# ==========================================
# スクリーンショットにある「マイドライブ > VK-MAP」の絶対パス
IMAGE_DIR = "/content/drive/MyDrive/VK-MAP"

# 生成するファイルの出力先（Drive内のVK-MAPフォルダに直接保存）
OUTPUT_INDEX_PATH = "/content/drive/MyDrive/VK-MAP/kofun_faiss.index"
OUTPUT_MAPPING_PATH = "/content/drive/MyDrive/VK-MAP/kofun_mapping.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ==========================================
# 3. DINOv2 モデルの準備
# ==========================================
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

print("DINOv2 モデルをロード中...")
model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
).to(device)
model.eval()

# ==========================================
# 4. 画像の収集と特徴量抽出
# ==========================================
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
  for file in files:
    if file.lower().endswith(VALID_EXTS):
      image_paths.append(os.path.join(root, file))

image_paths.sort()
print(f"検出された画像数: {len(image_paths)} 枚")

if len(image_paths) == 0:
  raise ValueError(
      f"指定されたフォルダ '{IMAGE_DIR}' 内に画像が見つかりませんでした。"
  )

vectors = []
index_to_kofun = {}

print("特徴量ベクトルの抽出を開始します...")
with torch.no_grad():
  for current_idx, img_path in enumerate(tqdm(image_paths)):
    try:
      img = Image.open(img_path).convert("RGB")
      input_tensor = transform(img).unsqueeze(0).to(device)

      feat_vec = model(input_tensor)
      feat_vec = feat_vec / feat_vec.norm(p=2, dim=-1, keepdim=True)
      vec_np = feat_vec.cpu().numpy().astype("float32").flatten()

      # パス構造から古墳名（または画像が属するフォルダ名/ファイル名）を取得
      # 例: VK-MAP/Fukukuoka/井寺古墳/01.jpg -> "井寺古墳"
      rel_path = os.path.relpath(img_path, IMAGE_DIR)
      parent_dir = os.path.dirname(rel_path)

      if parent_dir:
        # 最深部のフォルダ名を識別名とする（階層に応じてお好みで調整可能）
        kofun_name = os.path.basename(parent_dir)
      else:
        kofun_name = os.path.splitext(os.path.basename(img_path))[0]

      vectors.append(vec_np)
      index_to_kofun[current_idx] = {
          "kofun_name": kofun_name,
          "img_path": img_path,
          "filename": os.path.basename(img_path),
      }

    except Exception as e:
      print(f"\n⚠️ 読み込みエラー ({img_path}): {e}")

# ==========================================
# 5. FAISS インデックスの生成と保存
# ==========================================
vectors_np = np.array(vectors).astype("float32")
dimension = vectors_np.shape[1]

print(f"\nFAISS インデックス構築中 (次元数: {dimension})...")
index = faiss.IndexFlatIP(dimension)
index.add(vectors_np)

# Google Drive 上に直接保存
faiss.write_index(index, OUTPUT_INDEX_PATH)
with open(OUTPUT_MAPPING_PATH, "wb") as f:
  pickle.dump(index_to_kofun, f)

print("🎉 処理が完了しました！")
print(f"・インデックス: {OUTPUT_INDEX_PATH}")
print(f"・マッピング: {OUTPUT_MAPPING_PATH}")
print(f"・登録データ数: {index.ntotal} 件")
