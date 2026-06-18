# Face detector files

This folder holds OpenCV's pre-trained SSD face detector. The two required
files are **not** committed to git (the `.caffemodel` is a ~10 MB binary).

Get them by running, from the project root:

```bash
python download_models.py
```

That downloads:

- `deploy.prototxt` — the network architecture
- `res10_300x300_ssd_iter_140000.caffemodel` — the trained weights

Once both files are present here, `detect_mask_video.py` and
`detect_mask_image.py` will work.
