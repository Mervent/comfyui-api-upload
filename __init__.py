try:
    from .comfyui_api_upload.nodes import APIUpload, DownloadImage
except ImportError:
    from comfyui_api_upload.nodes import APIUpload, DownloadImage

NODE_CLASS_MAPPINGS = {
    "APIUpload": APIUpload,
    "DownloadImage": DownloadImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIUpload": "Upload Image",
    "DownloadImage": "Download Image",
}
