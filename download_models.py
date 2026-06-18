"""Download the OpenCV SSD face detector files into the face_detector/ folder.

The webcam / image detectors need two files that are too large (or just not
worth) committing to git:

    face_detector/deploy.prototxt                       (network definition)
    face_detector/res10_300x300_ssd_iter_140000.caffemodel  (trained weights)

Run this once after cloning:

    python download_models.py
"""
import os
import urllib.request

OUT_DIR = "face_detector"

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/chandrikadeb7/"
        "Face-Mask-Detection/master/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/chandrikadeb7/"
        "Face-Mask-Detection/master/face_detector/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(OUT_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print("[INFO] {} already exists, skipping.".format(name))
            continue
        print("[INFO] downloading {} ...".format(name))
        urllib.request.urlretrieve(url, dest)
        print("[INFO] saved to {}".format(dest))
    print("[INFO] done. Face detector is ready.")


if __name__ == "__main__":
    main()
