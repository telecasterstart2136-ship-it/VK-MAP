import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# PyTorchのメモリ割り当て最適化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from datetime import datetime
import gc
import pickle
import shutil
import faiss
import gdown
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st
import timm
import torch
from torchvision import transforms

# CPU環境でのスレッド数を1にしてメモリとCPUの過負荷を防ぐ
torch.set_num_threads(1)

# --------------------------------------------------
# Base Directory Configuration
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Google Drive 各ファイル ID 設定
# --------------------------------------------------
INDEX_FILE_ID = "1YgR1F2kPyVTqeB1cA9HAmKyMTzp5ZW8G"
MAPPING_FILE_ID = "1PXdUKwo6bFlNp1E7Qm0-UzVFZ_BcA0sA"


def fetch_from_drive(file_id, save_path):
    try:
        gdown.download(
            id=str(file_id), output=save_path, quiet=True, fuzzy=True
        )
    except Exception:
        pass

    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True

    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(url, stream=True, timeout=30)
                break

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
    except Exception:
        pass

    return os.path.exists(save_path) and os.path.getsize(save_path) > 1000


def download_index_files(target_dir):
  index_path = os.path.join(target_dir, "kofun_faiss.index")
  mapping_path = os.path.join(target_dir, "kofun_mapping.pkl")

  # 1MB 以下のファイルは破損（HTML等）とみなす判定サイズ閾値
  MIN_SIZE = 1024 * 1024

  os.makedirs(target_dir, exist_ok=True)

  with st.spinner("📦 Downloading index files..."):
    # インデックスファイルの確認と取得
    if not os.path.exists(index_path) or os.path.getsize(index_path) < MIN_SIZE:
      if os.path.exists(index_path):
        os.remove(index_path)
      if not fetch_from_drive(INDEX_FILE_ID, index_path):
        st.error(
            "⚠️ index ファイルの取得に失敗しました。Google Drive"
            " のアクセス権限（リンクを知っている全員）を確認してください。"
        )
        st.stop()

    # マッピングファイルの確認と取得
    if (
        not os.path.exists(mapping_path)
        or os.path.getsize(mapping_path) < 1000
    ):
      if os.path.exists(mapping_path):
        os.remove(mapping_path)
      if not fetch_from_drive(MAPPING_FILE_ID, mapping_path):
        st.error("⚠️ mapping ファイルの取得に失敗しました。")
        st.stop()

  return index_path, mapping_path


@st.cache_resource
def load_system():
  gc.collect()
  device = torch.device("cpu")

  transform = transforms.Compose([
      transforms.Resize((384, 384)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  model = timm.create_model(
      "convnext_nano", pretrained=True, num_classes=0
  ).to(device)
  model.eval()

  # キャッシュフォルダ名を変更して過去の破損ファイルを確実に回避
  cache_dir = os.path.join(BASE_DIR, "cache_vkmap_v2")
  index_file, mapping_file = download_index_files(cache_dir)

  index = faiss.read_index(index_file)
  with open(mapping_file, "rb") as f:
    index_to_kofun = pickle.load(f)

  return model, index, index_to_kofun, transform, device
    os.makedirs(target_dir, exist_ok=True)

  with st.spinner("📦 Downloading index files..."):
      if not os.path.exists(index_path) or os.path.getsize(index_path) <= 1000:
          if not fetch_from_drive(INDEX_FILE_ID, index_path):
                st.error("⚠️ index ファイルの取得に失敗しました。")
                st.stop()

        if (
            not os.path.exists(mapping_path)
            or os.path.getsize(mapping_path) <= 1000
        ):
            if not fetch_from_drive(MAPPING_FILE_ID, mapping_path):
                st.error("⚠️ mapping ファイルの取得に失敗しました。")
                st.stop()

    return index_path, mapping_path


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="VK-MAP (UI2)", layout="wide")
st.title("🏛️ VK-MAP (UI2)")
st.caption("Visual Kofun Matching System — Memory Optimized")


# --------------------------------------------------
# 2. Model & Cache Initialization (Low Memory)
# --------------------------------------------------
@st.cache_resource
def load_system():
    # ガベージコレクションで余分なメモリを解放
    gc.collect()

    device = torch.device("cpu")

    # 画像サイズを 384 または 224 に抑えてメモリ節約
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    # FAISSインデックスの次元数（768次元等）に合わせつつメモリオーバーを防ぐ
    model = timm.create_model(
        "convnext_nano", pretrained=True, num_classes=0
    ).to(device)
    model.eval()

    cache_dir = os.path.join(BASE_DIR, "cache_vkmap")
    index_file, mapping_file = download_index_files(cache_dir)

    index = faiss.read_index(index_file)
    with open(mapping_file, "rb") as f:
        index_to_kofun = pickle.load(f)

    return model, index, index_to_kofun, transform, device


with st.spinner("📦 Initializing system..."):
    model, index, index_to_kofun, transform, device = load_system()

st.success(f"✅ System Ready ({len(index_to_kofun)} features loaded)")

# --------------------------------------------------
# 3. UI & Match Handling
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Target Image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    query_tensor = transform(query_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_vec = model(query_tensor)
        query_vec = query_vec / query_vec.norm(p=2, dim=-1, keepdim=True)
        query_vec_np = query_vec.numpy().astype("float32")

    k_search = min(3, len(index_to_kofun))
    distances, indices = index.search(query_vec_np, k=k_search)

    top_score = float(distances[0][0])
    top_match = index_to_kofun[indices[0][0]]

    st.write(f"**Top Match:** {top_match['kofun_name']}")
    st.write(f"**Similarity Score:** {top_score:.4f}")

    # 使用済みオブジェクトの明示的破棄
    del query_tensor, query_vec, query_vec_np
    gc.collect()
