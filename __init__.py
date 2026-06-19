try:
    from .comfyui_api_upload.nodes import APIUpload
except ImportError:
    from comfyui_api_upload.nodes import APIUpload

NODE_CLASS_MAPPINGS = {
    "APIUpload": APIUpload,
}
