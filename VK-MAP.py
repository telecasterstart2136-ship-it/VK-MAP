import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
import pickle
import shutil
import faiss
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import timm
import torch
from torchvision import transforms

# CPU環境での並列演算スレッド数を最適化（高速化）
torch.set_num_threads(2)

# --------------------------------------------------
# Base Directory Configuration
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Google Drive の 各ファイル ID を設定
# （取得した実際のファイルIDに書き換えてください）
# --------------------------------------------------
INDEX_FILE_ID = "1ZKlD7uHexASfGBsyKIzNAtVC83f2xS4m"
MAPPING_FILE_ID ="1ZKlD7uHexASfGBsyKIzNAtVC83f2xS4m"


def download_index_files(target_dir):
  """Google Driveから index と mapping ファイルを個別に自動取得する関数"""
  index_path = os.path.join(target_dir, "kofun_faiss.index")
  mapping_path = os.path.join(target_dir, "kofun_mapping.pkl")

  # 既に正常なファイルが存在する場合はダウンロードをスキップ
  if (
      os.path.exists(index_path)
      and os.path.exists(mapping_path)
      and os.path.getsize(index_path) > 1000
  ):
    return index_path, mapping_path

  os.makedirs(target_dir, exist_ok=True)

  with st.spinner("📦 Downloading index files from Google Drive..."):
    # kofun_faiss.index の取得
    if not os.path.exists(index_path) or os.path.getsize(index_path) <= 1000:
      url_index = f"https://drive.google.com/uc?id={INDEX_FILE_ID}"
      gdown.download(
          url_index, index_path, quiet=False, fuzzy=True, use_cookies=False
      )

    # kofun_mapping.pkl の取得
    if (
        not os.path.exists(mapping_path)
        or os.path.getsize(mapping_path) <= 1000
    ):
      url_mapping = f"https://drive.google.com/uc?id={MAPPING_FILE_ID}"
      gdown.download(
          url_mapping, mapping_path, quiet=False, fuzzy=True, use_cookies=False
      )

  # ダウンロード後の検証
  if not os.path.exists(index_path) or not os.path.exists(mapping_path):
    st.error("⚠️ Google Drive からのファイルダウンロードに失敗しました。")
    st.stop()

  # HTML（アクセス制限エラー画面）が落ちていないか検証
  with open(index_path, "rb") as f:
    if b"<html" in f.read(100).lower():
      shutil.rmtree(target_dir, ignore_errors=True)
      st.error(
          "⚠️ Google Drive ファイルの取得に失敗しました。"
          " ファイル共有設定が「リンクを知っている全員」になっているか確認してください。"
      )
      st.stop()

  return index_path, mapping_path


# --------------------------------------------------
# Helper Functions for Path Resolution
# --------------------------------------------------
def resolve_path(rel_or_abs_path):
  if os.path.isabs(rel_or_abs_path):
    return rel_or_abs_path
  return os.path.join(BASE_DIR, rel_or_abs_path)


def find_valid_image_path(original_path, ref_dir_abs):
  if os.path.exists(original_path):
    return original_path

  filename = os.path.basename(original_path)
  for root, _, files in os.walk(ref_dir_abs):
    if filename in files:
      return os.path.join(root, filename)

  return None


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="VK-MAP (UI2)", layout="wide")
st.title("🏛️ VK-MAP (UI2)")
st.caption(
    "Visual Kofun Matching and Feature Profiling System — Automatic Database"
    " Matching"
)

# --------------------------------------------------
# 2. Sidebar Settings
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
)
reference_dir = st.sidebar.text_input(
    "Reference Data Directory", value="reference_data"
)


# --------------------------------------------------
# 3. Model & Cache Initialization
# --------------------------------------------------
@st.cache_resource
def load_system():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform = transforms.Compose([
      transforms.Resize((518, 518)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  model = timm.create_model(
      "convnext_small.fb_in22k_ft_in1k_384", pretrained=True, num_classes=0
  ).to(device)
  model.eval()

  cache_dir = os.path.join(BASE_DIR, "cache_vkmap_convnext")

  # 個別ファイルダウンロード関数を呼び出し
  index_file, mapping_file = download_index_files(cache_dir)

  # ロード処理
  index = faiss.read_index(index_file)
  with open(mapping_file, "rb") as f:
    index_to_kofun = pickle.load(f)

  return model, index, index_to_kofun, transform, device


with st.spinner("📦 Initializing ConvNeXt model and index..."):
  model, index, index_to_kofun, transform, device = load_system()

st.success(f"✅ System Ready ({len(index_to_kofun)} features loaded)")


# --------------------------------------------------
# 4. UI: File Upload Section
# --------------------------------------------------
st.subheader("1. Upload Target Image")
uploaded_file = st.file_uploader(
    "Drag and drop decorated pattern image here",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
  query_img = Image.open(uploaded_file).convert("RGB")
  query_tensor = transform(query_img).unsqueeze(0).to(device)

  with torch.inference_mode():
    query_vec = model(query_tensor)
    query_vec = query_vec / query_vec.norm(p=2, dim=-1, keepdim=True)
    query_vec_np = query_vec.cpu().numpy().astype("float32")

  k_search = min(3, len(index_to_kofun))
  distances, indices = index.search(query_vec_np, k=k_search)

  top_score = float(distances[0][0])
  top_match = index_to_kofun[indices[0][0]]
  predicted_label = (
      top_match["kofun_name"]
      if top_score >= threshold
      else "Unregistered (Low Similarity)"
  )

  rank2_match = index_to_kofun[indices[0][1]] if k_search > 1 else top_match
  rank2_score = float(distances[0][1]) if k_search > 1 else top_score

  rank3_match = index_to_kofun[indices[0][2]] if k_search > 2 else top_match
  rank3_score = float(distances[0][2]) if k_search > 2 else top_score

  # --------------------------------------------------
  # 5. UI: Prediction Results Table
  # --------------------------------------------------
  st.markdown("---")
  st.subheader("2. Matching Results")

  result_data = [{
      "Input File": uploaded_file.name,
      "Predicted Kofun": predicted_label,
      "Top Similarity": round(top_score, 4),
      "Rank 1 Match": top_match["kofun_name"],
      "Rank 2 Match": rank2_match["kofun_name"],
      "Rank 2 Score": round(rank2_score, 4),
      "Rank 3 Match": rank3_match["kofun_name"],
      "Rank 3 Score": round(rank3_score, 4),
  }]
  df_result = pd.DataFrame(result_data)

  m1, m2 = st.columns(2)
  m1.metric("Predicted Label", predicted_label)
  m2.metric("Top Similarity Score", f"{top_score:.4f}")

  st.dataframe(df_result, use_container_width=True)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
  st.download_button(
      label="📥 Download Result CSV",
      data=csv_bytes,
      file_name=f"VK-MAP_matching_result_{timestamp}.csv",
      mime="text/csv",
  )

  # --------------------------------------------------
  # 6. UI: Image Comparison
  # --------------------------------------------------
  st.markdown("---")
  st.subheader("3. Image Comparison (Target vs. Rank 1 Database Match)")

  ref_dir_abs = resolve_path(reference_dir)
  ref_img_path = find_valid_image_path(top_match["img_path"], ref_dir_abs)

  c1, c2 = st.columns(2)
  with c1:
    st.markdown("### 📷 Target Image")
    st.image(query_img, use_container_width=True)

  with c2:
    st.markdown(f"### 🖼️ Database Match (Top 1: {top_match['kofun_name']})")
    if ref_img_path and os.path.exists(ref_img_path):
      ref_img = Image.open(ref_img_path).convert("RGB")
      st.image(
          ref_img,
          caption=f"File: {os.path.basename(ref_img_path)}",
          use_container_width=True,
      )
    else:
      st.warning(f"⚠️ Reference image file not found: `{top_match['img_path']}`")

else:
  st.info("👆 Upload an image to search the reference database.")
