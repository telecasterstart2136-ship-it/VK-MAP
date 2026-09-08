import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
import pickle
import urllib.request  # ← この行を追加します！
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import timm
import torch
from torchvision import transforms
# --------------------------------------------------
# Base Directory Configuration (Resolves Read-Only & Absolute Path Issues)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_URL = "https://drive.google.com/drive/u/2/folders/1ZKlD7uHexASfGBsyKIzNAtVC83f2xS4m"
MAPPING_URL = "https://drive.google.com/drive/u/2/folders/1ZKlD7uHexASfGBsyKIzNAtVC83f2xS4m"


def download_file_from_cloud(url, save_path):
  """クラウドからファイルをダウンロードする関数"""
  if not os.path.exists(save_path):
    with st.spinner(
        f"Downloading {os.path.basename(save_path)} from cloud..."
    ):
      urllib.request.urlretrieve(url, save_path)


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
# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="VK-MAP (UI2)", layout="wide")
st.title("🏛️ VK-MAP (UI2)")
st.caption(
    "Visual Kofun Matching and Attention Profiling System — Automatic"
    " Database Matching"
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

rebuild_db = st.sidebar.button("🔄 Rebuild Feature Database")


# --------------------------------------------------
# Helper Functions for Path Resolution
# --------------------------------------------------
def resolve_path(rel_or_abs_path):
  """Convert relative paths to script-relative absolute paths."""
  if os.path.isabs(rel_or_abs_path):
    return rel_or_abs_path
  return os.path.join(BASE_DIR, rel_or_abs_path)


def find_valid_image_path(original_path, ref_dir_abs):
  """Fallback mechanism to resolve file path mismatches caused by folder/file renaming."""
  if os.path.exists(original_path):
    return original_path

  filename = os.path.basename(original_path)
  for root, _, files in os.walk(ref_dir_abs):
    if filename in files:
      return os.path.join(root, filename)

  return None


# --------------------------------------------------
# 3. Model & Cache Automatic Build Initialization
# --------------------------------------------------
@st.cache_resource
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

    # DINOv2 モデルの読み込み
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    ).to(device)
    model.eval()  # ← ここの先頭スペースを上の行（model = ...）とぴったり揃えます

    cache_dir = os.path.join(BASE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)


    index_file = os.path.join(cache_dir, "kofun_faiss.index")
    mapping_file = os.path.join(cache_dir, "kofun_mapping.pkl")

  # 1. キャッシュがローカルになければクラウドから高速ダウンロード
    if not (os.path.exists(index_file) and os.path.exists(mapping_file)):
      download_file_from_cloud(INDEX_URL, index_file)
      download_file_from_cloud(MAPPING_URL, mapping_file)

  # 2. クラウドから取得した（または既存の）インデックスをロード
      index = faiss.read_index(index_file)
      with open(mapping_file, "rb") as f:
         index_to_kofun = pickle.load(f)

      return model, index, index_to_kofun, transform, device


with st.spinner("📦 Initializing DINOv2 model and index..."):
  model, index, index_to_kofun, transform, device = load_system()

st.success(f"✅ System Ready ({len(index_to_kofun)} features loaded)")

# --------------------------------------------------
# 4. Attention Map Generator
# --------------------------------------------------
def generate_heatmap_fig(img_pil, input_tensor, model, title=""):
  patch_size = 14
  w, h = input_tensor.shape[2], input_tensor.shape[3]
  w_featmap, h_featmap = w // patch_size, h // patch_size

  attentions = None

  def hook_fn(module, input, output):
    nonlocal attentions
    attentions = output

  handle = model.blocks[-1].attn.qkv.register_forward_hook(hook_fn)
  with torch.no_grad():
    _ = model(input_tensor)
  handle.remove()

  if attentions is None:
    return None

  B, N, C = attentions.shape
  qkv = (
      attentions.reshape(
          B,
          N,
          3,
          model.blocks[-1].attn.num_heads,
          C // (3 * model.blocks[-1].attn.num_heads),
      )
      .permute(2, 0, 3, 1, 4)
  )
  q, k = qkv[0], qkv[1]

  scale = (C // (3 * model.blocks[-1].attn.num_heads)) ** -0.5
  attn = (q @ k.transpose(-2, -1)) * scale
  attn = attn.softmax(dim=-1)

  cls_attn = (
      attn[0, :, 0, 1:]
      .mean(dim=0)
      .reshape(w_featmap, h_featmap)
      .cpu()
      .numpy()
  )
  cls_attn_resized = np.array(
      Image.fromarray(cls_attn).resize(img_pil.size, Image.BICUBIC)
  )
  cls_attn_norm = (cls_attn_resized - cls_attn_resized.min()) / (
      cls_attn_resized.max() - cls_attn_resized.min() + 1e-8
  )

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.imshow(img_pil)
  ax.imshow(cls_attn_norm, cmap="jet", alpha=0.5)
  ax.set_title(title, fontsize=10)
  ax.axis("off")
  plt.tight_layout()
  return fig


# --------------------------------------------------
# 5. UI: File Upload Section
# --------------------------------------------------
st.subheader("1. Upload Target Image")
uploaded_file = st.file_uploader(
    "Drag and drop decorated pattern image here",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
  query_img = Image.open(uploaded_file).convert("RGB")
  query_tensor = transform(query_img).unsqueeze(0).to(device)

  # Search in Database
  with torch.no_grad():
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
  # 6. UI: Prediction Results Table
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

  # CSV Download Button
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
  st.download_button(
      label="📥 Download Result CSV",
      data=csv_bytes,
      file_name=f"VK-MAP_matching_result_{timestamp}.csv",
      mime="text/csv",
  )

  # --------------------------------------------------
  # 7. UI: Attention Heatmap Comparison
  # --------------------------------------------------
  st.markdown("---")
  st.subheader(
      "3. Attention Map Profiling (Target vs. Rank 1 Database Match)"
  )

  ref_img_path = find_valid_image_path(top_match["img_path"], ref_dir_abs)

  if ref_img_path and os.path.exists(ref_img_path):
    ref_img = Image.open(ref_img_path).convert("RGB")
    ref_tensor = transform(ref_img).unsqueeze(0).to(device)

    with st.spinner("Generating attention heatmaps..."):
      fig_query = generate_heatmap_fig(
          query_img,
          query_tensor,
          model,
          title=f"Target: {uploaded_file.name}",
      )
      fig_ref = generate_heatmap_fig(
          ref_img,
          ref_tensor,
          model,
          title=f"Top 1 Match: {top_match['kofun_name']}",
      )

    c1, c2 = st.columns(2)
    with c1:
      st.markdown("### 📷 Target Image")
      st.image(query_img, use_container_width=True)
      if fig_query:
        st.pyplot(fig_query)

    with c2:
      st.markdown(
          f"### 🖼️ Database Match (Top 1: {top_match['kofun_name']})"
      )
      st.image(
          ref_img,
          caption=f"File: {os.path.basename(ref_img_path)}",
          use_container_width=True,
      )
      if fig_ref:
        st.pyplot(fig_ref)
  else:
    st.error(f"⚠️ Reference image file not found: `{top_match['img_path']}`")

else:
  st.info(
      "👆 Upload an image to search the reference database and view attention"
      " map profiling."
  )
