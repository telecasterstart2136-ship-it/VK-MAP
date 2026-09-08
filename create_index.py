import glob
import os
import pickle
import faiss
import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------
# 設定
# --------------------------------------------------
# 参照画像が格納されているディレクトリのパス
IMAGE_DIR = "reference_data"

# 出力ファイル名
OUTPUT_INDEX_FILE = "kofun_faiss.index"
OUTPUT_MAPPING_FILE = "kofun_mapping.pkl"

# 画像の対応拡張子
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"⚡ 使用デバイス: {device}")

  # 1. 518x518解像度の前処理設定
  transform = transforms.Compose([
      transforms.Resize((518, 518)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  # 2. ConvNeXt Small モデルの読み込み
  print("📦 ConvNeXt Small モデルを読み込み中...")
  model = timm.create_model(
      "convnext_small.fb_in22k_ft_in1k_384", pretrained=True, num_classes=0
  ).to(device)
  model.eval()

  # 3. 画像ファイルの収集
  image_paths = []
  for ext in IMAGE_EXTENSIONS:
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True)
    )

  image_paths = sorted(list(set(image_paths)))
  print(f"🖼️ 対象画像数: {len(image_paths)} 枚")

  if len(image_paths) == 0:
    print(
        f"❌ エラー: '{IMAGE_DIR}' ディレクトリ内に画像が見つかりませんでした。"
    )
    return

  # 4. 特徴量抽出
  features_list = []
  mapping_dict = {}

  print("🚀 特徴量抽出を開始します...")
  with torch.inference_mode():
    for idx, img_path in enumerate(tqdm(image_paths)):
      try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # 特徴抽出 ＆ L2正規化
        feat = model(tensor)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy().astype("float32").flatten()

        features_list.append(feat_np)

        # マッピング情報の登録（ファイル名から古墳名を抽出するロジックは必要に応じて調整）
        filename = os.path.basename(img_path)
        kofun_name = os.path.splitext(filename)[0]

        mapping_dict[idx] = {
            "img_path": img_path,
            "kofun_name": kofun_name,
            "filename": filename,
        }

      except Exception as e:
        print(f"\n⚠️ スキップ ({img_path}): {e}")

  # 5. FAISS インデックスの構築 (コサイン類似度用 FlatIP)
  features_array = np.array(features_list).astype("float32")
  dimension = features_array.shape[1]

  print(
      f"\n📊 FAISSインデックス構築中... (次元数: {dimension}, データ数:"
      f" {len(features_array)})"
  )
  index = faiss.IndexFlatIP(dimension)  # L2正規化済みベクトルの内積＝コサイン類似度
  index.add(features_array)

  # 6. 保存
  faiss.write_index(index, OUTPUT_INDEX_FILE)
  with open(OUTPUT_MAPPING_FILE, "wb") as f:
    pickle.dump(mapping_dict, f)

  print("✅ 完了しました！")
  print(f" - {OUTPUT_INDEX_FILE}")
  print(f" - {OUTPUT_MAPPING_FILE}")
  print(
      "\n上記の2ファイルを Google Drive の 'VK-MAP' フォルダへアップロードしてください。"
  )


if __name__ == "__main__":
  main()
