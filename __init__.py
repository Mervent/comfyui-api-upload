try:
    from .comfyui_api_upload.nodes import DownloadImage, UploadImage, UploadVideo
except ImportError:
    from comfyui_api_upload.nodes import DownloadImage, UploadImage, UploadVideo

NODE_CLASS_MAPPINGS = {
    "UploadImage": UploadImage,
    "DownloadImage": DownloadImage,
    "UploadVideo": UploadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadImage": "Upload Image",
    "DownloadImage": "Download Image",
    "UploadVideo": "Upload Video",
}
