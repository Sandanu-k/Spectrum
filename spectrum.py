!apt-get update -qq
!apt-get install -y -qq colmap
!pip install -q open3d matplotlib opencv-python pillow tqdm

import os, sys, glob, subprocess
from PIL import Image
from tqdm import tqdm



RAW_DIR    = "/kaggle/input/datasets/samarthguptha/copy-of-lighthouse-survey-db4"
WORKSPACE  = "/kaggle/working/lighthouse_workspace_fast"
IMAGE_DIR  = f"{WORKSPACE}/images"
DB_PATH    = f"{WORKSPACE}/database.db"
SPARSE_DIR = f"{WORKSPACE}/sparse"
VOCAB_PATH = "/kaggle/working/vocab_tree_flickr100K_words32K.bin"

for d in [WORKSPACE, IMAGE_DIR, SPARSE_DIR]:
    os.makedirs(d, exist_ok=True)

NUM_THREADS = 4

def run_cmd(title, command):
    print(f"\n{'='*60}\n[*] {title}\n{'='*60}")
    command = f"stdbuf -oL -eL {command}"
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Step failed: {title}")
    print(f"[✓] {title} done")
raw_images = sorted(glob.glob(f"{RAW_DIR}/**/*.jpg", recursive=True))
print(f"Found {len(raw_images)} raw images")

print("Downscaling images...")
for img_path in tqdm(raw_images):
    dest = os.path.join(IMAGE_DIR, os.path.basename(img_path))
    with Image.open(img_path) as img:
        exif_data = img.info.get('exif')
        img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        if exif_data:
            img.save(dest, "JPEG", quality=92, exif=exif_data)
        else:
            img.save(dest, "JPEG", quality=92)

print(f"[✓] {len(os.listdir(IMAGE_DIR))} images ready")

if not os.path.exists(VOCAB_PATH):
    print("Downloading vocab tree...")
    subprocess.run(f"wget -q https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin -O {VOCAB_PATH}", shell=True, check=True)
print(f"[✓] Vocab tree ready at {VOCAB_PATH}")

run_cmd(
    "SIFT Feature Extraction",
    f"colmap feature_extractor --database_path {DB_PATH} --image_path {IMAGE_DIR} "
    f"--ImageReader.single_camera 1 --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads {NUM_THREADS}"
)


run_cmd(
    "Vocab Tree Matching",
    f"colmap vocab_tree_matcher --database_path {DB_PATH} "
    f"--VocabTreeMatching.vocab_tree_path {VOCAB_PATH} "
    f"--VocabTreeMatching.num_images 20 "
    f"--SiftMatching.use_gpu 0 --SiftMatching.num_threads {NUM_THREADS}"
)


run_cmd(
    "Structure from Motion (SfM)",
    f"colmap mapper --database_path {DB_PATH} --image_path {IMAGE_DIR} --output_path {SPARSE_DIR} "
    f"--Mapper.ba_global_images_ratio 1.2 --Mapper.ba_global_points_ratio 1.2 --Mapper.num_threads {NUM_THREADS}"
)

print("\n SfM complete. Checking registration quality:")
!colmap model_analyzer --path {SPARSE_DIR}/0v


!ls -la {SPARSE_DIR}

!colmap model_analyzer --path {SPARSE_DIR}/0

run_cmd(
    "Export sparse model to PLY",
    f"colmap model_converter --input_path {SPARSE_DIR}/0 --output_path {WORKSPACE}/sparse_point_cloud.ply --output_type PLY"
)
